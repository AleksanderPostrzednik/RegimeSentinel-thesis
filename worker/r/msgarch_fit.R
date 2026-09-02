#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default = NULL) {
  position <- match(paste0("--", name), args)
  if (is.na(position) || position == length(args)) return(default)
  args[[position + 1]]
}

emit <- function(key, ...) {
  values <- c(key, list(...))
  cat(paste(vapply(values, as.character, character(1)), collapse = "\t"), "\n", sep = "")
}

num <- function(value) sprintf("%.17g", as.numeric(value))

data_path <- get_arg("data")
mode <- get_arg("mode")
start_index <- as.integer(get_arg("start-index", "0"))
seed <- as.integer(get_arg("seed", "20260722"))
par0_path <- get_arg("par0")
if (!(mode %in% c("preflight", "rolling"))) {
  emit("STATUS", "error")
  emit("ERROR", "mode must be preflight or rolling")
  quit(save = "no", status = 0)
}
if (is.null(data_path) || !file.exists(data_path)) {
  emit("STATUS", "error")
  emit("ERROR", "data file is missing")
  quit(save = "no", status = 0)
}

values <- tryCatch(scan(data_path, quiet = TRUE), error = function(error) NULL)
if (is.null(values) || length(values) != 500 || any(!is.finite(values))) {
  emit("STATUS", "error")
  emit("ERROR", "data must contain exactly 500 finite returns")
  quit(save = "no", status = 0)
}

if (!requireNamespace("MSGARCH", quietly = TRUE)) {
  emit("STATUS", "error")
  emit("ERROR", "R package MSGARCH is unavailable")
  quit(save = "no", status = 0)
}

effective_seed <- seed + start_index
set.seed(effective_seed)
spec <- tryCatch(
  MSGARCH::CreateSpec(
    variance.spec = list(model = c("sGARCH", "sGARCH")),
    distribution.spec = list(distribution = c("std", "std")),
    switch.spec = list(do.mix = FALSE),
    constraint.spec = list(regime.const = c("nu"))
  ),
  error = function(error) error
)
if (inherits(spec, "error")) {
  emit("STATUS", "error")
  emit("ERROR", conditionMessage(spec))
  quit(save = "no", status = 0)
}

base <- spec$par0
labels <- names(base)
if (is.null(labels)) {
  emit("STATUS", "error")
  emit("ERROR", "MSGARCH spec returned unnamed parameters")
  quit(save = "no", status = 0)
}

start_source <- "deterministic_grid"
if (!is.null(par0_path) && file.exists(par0_path)) {
  start_source <- "provided_par0"
  start <- tryCatch(scan(par0_path, quiet = TRUE), error = function(error) NULL)
  if (is.null(start) || length(start) != length(base)) {
    emit("STATUS", "error")
    emit("ERROR", "provided par0 has an invalid length")
    quit(save = "no", status = 0)
  }
  names(start) <- labels
} else {
  start <- base
  alpha1_values <- c(0.04, 0.08, 0.14, 0.02, 0.10)
  beta1_values <- c(0.90, 0.82, 0.72, 0.94, 0.86)
  alpha2_values <- c(0.18, 0.10, 0.06, 0.24, 0.12)
  beta2_values <- c(0.70, 0.84, 0.90, 0.64, 0.80)
  omega_multiplier <- c(0.6, 1.0, 1.8, 0.8, 1.3)
  persistence_values <- c(0.80, 0.70, 0.90, 0.60, 0.85)
  for (state in c(1, 2)) {
    alpha_name <- paste0("alpha1_", state)
    beta_name <- paste0("beta_", state)
    omega_name <- paste0("alpha0_", state)
    if (alpha_name %in% labels) {
      start[[alpha_name]] <- if (state == 1) alpha1_values[[start_index + 1]] else alpha2_values[[start_index + 1]]
    }
    if (beta_name %in% labels) {
      start[[beta_name]] <- if (state == 1) beta1_values[[start_index + 1]] else beta2_values[[start_index + 1]]
    }
    if (omega_name %in% labels) {
      start[[omega_name]] <- base[[omega_name]] * omega_multiplier[[start_index + 1]] * ifelse(state == 1, 0.7, 1.4)
    }
  }
  nu_value <- c(5.0, 7.0, 10.0, 15.0, 8.0)[[start_index + 1]]
  for (nu_name in c("nu_1", "nu_2")) {
    if (nu_name %in% labels) start[[nu_name]] <- nu_value
  }
  persistence <- persistence_values[[start_index + 1]]
  if ("P_1_1" %in% labels) start[["P_1_1"]] <- persistence
  if ("P_2_1" %in% labels) start[["P_2_1"]] <- 1 - persistence
  lower <- spec$lower
  upper <- spec$upper
  for (i in seq_along(start)) {
    if (is.finite(lower[[i]])) start[[i]] <- max(start[[i]], lower[[i]] + 1e-8)
    if (is.finite(upper[[i]])) start[[i]] <- min(start[[i]], upper[[i]] - 1e-8)
  }
}

emit("MASTER_SEED", seed)
emit("EFFECTIVE_SEED", effective_seed)
emit("START_SOURCE", start_source)
for (i in seq_along(start)) emit("START_PARAM", names(start)[[i]], num(start[[i]]))

unmap_parameters <- getFromNamespace("f_unmapPar", "MSGARCH")
remove_regime_constant <- getFromNamespace("f_remove_regimeconstpar", "MSGARCH")
optimizer_start <- unmap_parameters(start, spec, TRUE)
if (isTRUE(spec[["regime.const.pars.bool"]])) {
  optimizer_start <- remove_regime_constant(
    optimizer_start,
    spec[["regime.const.pars"]],
    spec[["K"]],
    for.se = TRUE
  )
}

optimizer_capture <- new.env(parent = emptyenv())
optimizer_capture$result <- NULL
optimizer_capture$received <- NULL
optimizer_capture$used <- NULL
diagnostic_optim <- function(vPw, f_nll, spec, data, do.plm) {
  optimizer_capture$received <- vPw
  optimizer_capture$used <- optimizer_start
  result <- try(
    stats::optim(
      optimizer_start,
      f_nll,
      spec = spec,
      data = data,
      do.plm = do.plm,
      hessian = TRUE,
      method = "BFGS"
    ),
    silent = TRUE
  )
  optimizer_capture$result <- result
  result
}

fit_control <- list(par0 = start, do.se = FALSE, do.plm = TRUE)
if (identical(mode, "preflight")) fit_control$OptimFUN <- diagnostic_optim

fit <- tryCatch(
  MSGARCH::FitML(
    spec = spec,
    data = values,
    ctr = fit_control
  ),
  error = function(error) error
)
if (inherits(fit, "error")) {
  emit("STATUS", "error")
  emit("ERROR", conditionMessage(fit))
  quit(save = "no", status = 0)
}

if (identical(mode, "preflight")) {
  optimizer <- optimizer_capture$result
  if (!is.list(optimizer)) {
    emit("STATUS", "error")
    emit("ERROR", "MSGARCH optimizer diagnostics were unavailable")
    quit(save = "no", status = 0)
  }

  emit("OPTIM_METHOD", "BFGS")
  emit("OPTIM_DO_PLM", "true")
  emit("OPTIM_DO_SE", "false")
  emit("OPTIM_CONVERGENCE", optimizer$convergence)
  optimizer_message <- if (is.null(optimizer$message)) "" else gsub("[\\r\\n\\t]+", " ", optimizer$message)
  emit("OPTIM_MESSAGE", optimizer_message)
  emit("OPTIM_OBJECTIVE", num(optimizer$value))
  if (!is.null(optimizer$counts)) {
    for (count_name in names(optimizer$counts)) {
      emit("OPTIM_COUNT", count_name, optimizer$counts[[count_name]])
    }
  }
  emit_optimizer_vector <- function(key, values) {
    if (is.null(values)) return()
    value_names <- names(values)
    for (i in seq_along(values)) {
      value_name <- if (is.null(value_names)) "" else value_names[[i]]
      emit(key, i, value_name, num(values[[i]]))
    }
  }
  emit_optimizer_vector("OPTIM_START_RECEIVED", optimizer_capture$received)
  emit_optimizer_vector("OPTIM_START_USED", optimizer_capture$used)
  emit_optimizer_vector("OPTIM_END", optimizer$par)

  hessian <- optimizer$hessian
  hessian_available <- is.matrix(hessian) && nrow(hessian) == ncol(hessian) && all(is.finite(hessian))
  emit("HESSIAN_AVAILABLE", tolower(as.character(hessian_available)))
  if (hessian_available) {
    emit("HESSIAN_DIM", nrow(hessian))
    for (row in seq_len(nrow(hessian))) {
      for (column in seq_len(ncol(hessian))) {
        emit("HESSIAN", row, column, num(hessian[[row, column]]))
      }
    }
    hessian_eigenvalues <- tryCatch(
      eigen((hessian + t(hessian)) / 2, symmetric = TRUE, only.values = TRUE)$values,
      error = function(error) NULL
    )
    if (!is.null(hessian_eigenvalues) && all(is.finite(hessian_eigenvalues))) {
      for (i in seq_along(hessian_eigenvalues)) {
        emit("HESSIAN_EIGEN", i, num(hessian_eigenvalues[[i]]))
      }
    }
  }
}

par <- fit$par
if (is.null(names(par))) names(par) <- labels
transition <- tryCatch(MSGARCH::TransMat(fit), error = function(error) NULL)
state <- tryCatch(MSGARCH::State(fit), error = function(error) NULL)
risk <- tryCatch(MSGARCH::Risk(fit, alpha = c(0.05, 0.01), do.es = TRUE, nahead = 1), error = function(error) NULL)
if (is.null(transition) || is.null(state) || is.null(risk)) {
  emit("STATUS", "error")
  emit("ERROR", "MSGARCH fit did not expose transition, filtered state, and risk outputs")
  quit(save = "no", status = 0)
}

filtered <- drop(state$FiltProb)
if (length(dim(filtered)) == 3) filtered <- filtered[, 1, ]
if (is.null(dim(filtered))) filtered <- matrix(filtered, ncol = 2)
if (ncol(filtered) != 2) filtered <- t(filtered)
occupancy <- colMeans(filtered)
filtered_last <- filtered[nrow(filtered), ]

get_parameter <- function(parameter, state_index) {
  candidates <- c(paste0(parameter, "_", state_index), paste0(parameter, state_index))
  for (candidate in candidates) {
    if (candidate %in% names(par)) return(as.numeric(par[[candidate]]))
  }
  NA_real_
}

unc_vol <- c(NA_real_, NA_real_)
for (state_index in c(1, 2)) {
  alpha0 <- get_parameter("alpha0", state_index)
  alpha1 <- get_parameter("alpha1", state_index)
  beta <- get_parameter("beta", state_index)
  unc_vol[[state_index]] <- sqrt(alpha0 / (1 - alpha1 - beta))
}
state_order <- order(unc_vol)

emit("STATUS", "success")
emit("LOG_LIK", num(fit$loglik))
for (i in seq_along(par)) emit("PARAM", names(par)[[i]], num(par[[i]]))
for (row in c(1, 2)) for (column in c(1, 2)) emit("TRANS", row, column, num(transition[[row, column]]))
for (state_index in c(1, 2)) {
  emit("OCCUPANCY", state_index, num(occupancy[[state_index]]))
  emit("FILTERED_LAST", state_index, num(filtered_last[[state_index]]))
  emit("UNC_VOL", state_index, num(unc_vol[[state_index]]))
}
emit("STATE_ORDER", state_order[[1]], state_order[[2]])
for (i in seq_along(c(0.95, 0.99))) {
  confidence <- c(0.95, 0.99)[[i]]
  var_value <- -as.numeric(risk$VaR[1, i])
  es_value <- -as.numeric(risk$ES[1, i])
  emit("RISK", sprintf("%.2f", confidence), num(var_value), num(es_value))
}

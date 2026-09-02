#!/usr/bin/env bash
set -euo pipefail

rs_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
rs_repo_root="$(cd "${rs_script_dir}/../.." && pwd)"
rs_runtime_root="${REGIMESENTINEL_MSGARCH_RUNTIME:-${rs_repo_root}/.runtime/msgarch-r4.4.3-msgarch2.51}"
rs_micromamba="${rs_runtime_root}/bin/micromamba"
rs_mamba_root="${rs_runtime_root}/mamba-root"
rs_env="${rs_runtime_root}/env"
rs_sources="${rs_runtime_root}/sources"
rs_conda_lock="${rs_script_dir}/conda-linux-64.explicit.lock"

rs_micromamba_url="https://github.com/mamba-org/micromamba-releases/releases/download/2.9.0-0/micromamba-linux-64"
rs_micromamba_sha256="366cd9cd8be14df1ab8ed50352a82111082a36686b2d389fdb79a92c3fafb3e3"
rs_fanplot_url="https://cran.r-project.org/src/contrib/fanplot_4.0.1.tar.gz"
rs_fanplot_sha256="04e78587b29e4817b382b0abbbcad1b7e7a30aa789cb308ff5e8e74d58c4b2dd"
rs_msgarch_url="https://cran.r-project.org/src/contrib/MSGARCH_2.51.tar.gz"
rs_msgarch_sha256="b053e8a125a9fd0784aa496ea35021d2cfb3491210fb11554d3ca27b02481bd9"

download_verified() {
  local rs_url="$1"
  local rs_expected_sha256="$2"
  local rs_destination="$3"
  local rs_download="${rs_destination}.download"

  if printf '%s  %s\n' "${rs_expected_sha256}" "${rs_destination}" | sha256sum -c - >/dev/null 2>&1; then
    return
  fi
  curl -fL "${rs_url}" -o "${rs_download}"
  printf '%s  %s\n' "${rs_expected_sha256}" "${rs_download}" | sha256sum -c -
  mv -f "${rs_download}" "${rs_destination}"
}

install -d "${rs_runtime_root}/bin" "${rs_sources}"
download_verified "${rs_micromamba_url}" "${rs_micromamba_sha256}" "${rs_micromamba}"
chmod 0755 "${rs_micromamba}"

if [[ ! -x "${rs_env}/bin/Rscript" ]]; then
  MAMBA_ROOT_PREFIX="${rs_mamba_root}" "${rs_micromamba}" create -y -p "${rs_env}" --file "${rs_conda_lock}"
fi
"${rs_env}/bin/Rscript" -e 'stopifnot(as.character(getRversion()) == "4.4.3")'

rs_fanplot_tar="${rs_sources}/fanplot_4.0.1.tar.gz"
rs_msgarch_tar="${rs_sources}/MSGARCH_2.51.tar.gz"
download_verified "${rs_fanplot_url}" "${rs_fanplot_sha256}" "${rs_fanplot_tar}"
download_verified "${rs_msgarch_url}" "${rs_msgarch_sha256}" "${rs_msgarch_tar}"

if ! "${rs_env}/bin/Rscript" -e 'stopifnot(as.character(packageVersion("fanplot")) == "4.0.1")'; then
  MAMBA_ROOT_PREFIX="${rs_mamba_root}" "${rs_micromamba}" run -p "${rs_env}" R CMD INSTALL "${rs_fanplot_tar}"
fi
if ! "${rs_env}/bin/Rscript" -e 'stopifnot(as.character(packageVersion("MSGARCH")) == "2.51")'; then
  MAMBA_ROOT_PREFIX="${rs_mamba_root}" "${rs_micromamba}" run -p "${rs_env}" R CMD INSTALL "${rs_msgarch_tar}"
fi

"${rs_env}/bin/Rscript" -e 'stopifnot(requireNamespace("MSGARCH", quietly=TRUE)); cat(as.character(packageVersion("MSGARCH")), "\n")'
printf '%s\n' "${rs_env}/bin/Rscript"

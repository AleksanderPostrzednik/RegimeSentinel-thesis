# Odtworzenie analiz

## Zakres i kanoniczne wejścia

Zamrożony kontrakt eksperymentu `thesis-v1` znajduje się w:

- `experiments/thesis-v1/protocol.json`
- `data/snapshots/btc-eth-daily-close-2021-07-20_2026-07-19.json`

Snapshot obejmuje 1826 cen i 1825 log-zwrotów na instrument. Odtworzenie korzysta z dołączonego snapshotu i nie pobiera danych z sieci. Zmiana danych, okna, modelu, ryzyka, trybu preflight lub metryk wymaga nowego `protocolId`.

## Wersje środowiska

`worker/r/bootstrap_msgarch_env.sh` przypina i sprawdza R 4.4.3, MSGARCH 2.51 oraz fanplot 4.0.1. `worker/pyproject.toml` zawiera jedną wersję dokładną (`yfinance==0.2.65`) oraz kilka minimalnych zakresów (`numpy`, `scipy`, `arch`, `statsmodels`). Ścisła reprodukcja korzysta z wersji zapisanych w manifestach artefaktów i nie zakłada identyczności środowiska między maszynami.

Instalacja workera w nowym środowisku:

```bash
cd worker
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Instalacja środowiska R/MS-GARCH jest opcjonalna dla szybkiego audytu:

```bash
cd worker
./r/bootstrap_msgarch_env.sh
```

## Szybki audyt offline

Audyt nie uruchamia nowych modeli, nie pobiera danych i nie zastępuje pełnego przebiegu badania. Weryfikuje protokół oraz testy workera:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=worker/src \
python3 -m regime_sentinel_worker.experiment_protocol \
  experiments/thesis-v1/protocol.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=worker/src \
python3 -m unittest discover -s worker/tests -v
```

Pełny baseline i etap regime są kosztowne. Jeżeli wykonuje się rerun, każdy przebieg musi zapisywać się do nowego, pustego katalogu roboczego; nie należy używać istniejących katalogów artefaktów jako celu zapisu.

## Kanoniczne artefakty i interpretacja

- `artifacts/thesis-v1/baseline` — pełny baseline GARCH(1,1).
- `artifacts/thesis-v1/regime` — etap regime z fallbackiem oznaczonym `fallback_not_ms_garch`.
- `artifacts/thesis-v1/runs/true-msgarch-attempt-1-20260818T153348Z/` — pierwszy kontrolowany preflight MSGARCH.
- `artifacts/thesis-v1/runs/true-msgarch-attempt-2-20260818T174828Z/` — drugi i finalny kontrolowany preflight MSGARCH.
- `artifacts/thesis-v1/posthoc-diagnostics` — opisowe diagnostyki post-hoc.
- `artifacts/thesis-v1/provisional` — raport porównawczy bazowy i fallback bez deklarowania zwycięzcy.

`fallback_not_ms_garch` nie jest MS-GARCH. Brak rolling OOS MS-GARCH oznacza brak formalnie domkniętego porównania OOS, a nie dowód przegranej MS-GARCH.

## Mapa twierdzenie → artefakt

| Twierdzenie lub element | Artefakt źródłowy |
| --- | --- |
| Zamrożone reguły eksperymentu | `experiments/thesis-v1/protocol.json` |
| Zakres i pochodzenie danych wejściowych | `data/snapshots/btc-eth-daily-close-2021-07-20_2026-07-19.json` |
| Wyniki baseline GARCH(1,1) | `artifacts/thesis-v1/baseline` |
| Wynik etapu regime i fallback | `artifacts/thesis-v1/regime` |
| Dowód pierwszego preflight MSGARCH | `artifacts/thesis-v1/runs/true-msgarch-attempt-1-20260818T153348Z/` |
| Dowód drugiego preflight MSGARCH | `artifacts/thesis-v1/runs/true-msgarch-attempt-2-20260818T174828Z/` |
| Diagnostyki opisowe | `artifacts/thesis-v1/posthoc-diagnostics` |
| Raport porównawczy bez zwycięzcy | `artifacts/thesis-v1/provisional` |

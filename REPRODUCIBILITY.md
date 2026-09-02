# Odtworzenie analiz

## 1. Wymagania

Python:

- `>=3.10`
- `yfinance==0.2.65`
- `numpy>=1.24`
- `scipy>=1.10`
- `arch>=7.0`
- `statsmodels>=0.14`

R:

- `R 4.4.3`
- `MSGARCH 2.51`
- `fanplot 4.0.1` (wymagany przez środowisko MSGARCH)

Wariant replikacyjny oparty jest o ten sam kontrakt eksperymentu i te same binaria środowiska; inne wersje bibliotek to inny eksperyment.

## 2. Dane

Repozytorium zawiera zamrożony snapshot użyty w `thesis-v1`, bez pobierania, imputacji, forward-fill, interpolacji i cappingu:

- `data/snapshots/btc-eth-daily-close-2021-07-20_2026-07-19.json`

Wersja i liczba obserwacji wynikają z `protocol.json`:

- 1826 cen i 1825 log-zwrotów na instrument.

## 3. Protokół

Kontrakt eksperymentu to:

- `protocol/thesis-v1.json`

Każda zmiana tego pliku tworzy odrębny eksperyment (inny `protocolId`/warunki), zgodnie z `changeControl`:

- zmiana danych, okna, modelu, ryzyka, trybu preflight lub metryk wymaga nowego protokołu.

## 4. Instalacja (WSL/Linux)

Minimalna ścieżka instalacji:

```bash
cd worker
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

Środowisko R/MS-GARCH:

```bash
cd worker
./r/bootstrap_msgarch_env.sh
```

Skrypt buduje (lub używa ponownie) izolowane środowisko `R 4.4.3 + MSGARCH 2.51` zgodnie z `worker/r/runtime-sources.lock.json`.

## 5. Walidacja

Polecenia uruchamiające istniejące walidacje i testy:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=worker/src \
python3 -m regime_sentinel_worker.experiment_protocol protocol/thesis-v1.json
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=worker/src python3 -m unittest discover -s worker/tests -v
```

Pierwsza komenda sprawdza `protocol.json` (weryfikacja schematu + reguły domenowe), druga uruchamia dostępne testy jednostkowe.

## 6. Baseline GARCH

Baseline uruchamia się w nowym pustym katalogu roboczym. Nie uruchamiać nad istniejącym `artifacts/`.

```bash
cd worker
NEW_BASELINE_DIR=/tmp/regime-sentinel-baseline-run
mkdir -p "$NEW_BASELINE_DIR"
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m regime_sentinel_worker.main baseline \
  --artifacts "$NEW_BASELINE_DIR"
```

Wynik trafia do: `$NEW_BASELINE_DIR/baseline`.

## 7. MS-GARCH

W `msgarch-preflight` uruchamiany jest pipeline preflight dla obu instrumentów:

- pobiera pierwsze 500 obserwacji (okno `estimationWindowReturns: 500`),
- każdy instrument jest dopasowywany z **dokładnie 5 startami** (`start_index = 0..4`),
- dla każdego startu wywoływany jest `Rscript worker/r/msgarch_fit.R` z `--mode preflight`,
- dla każdego wywołania `FitML` używany jest domyślny optimizer `stats::optim` z metodą `BFGS`.

Próg bramki preflight (`run_preflight`) wymaga łącznie:

- 5 dopasowań na instrument,
- wszystkie dopasowania poprawne i ważne,
- skończona log-wiarygodność,
- spełnienie ograniczeń parametrów i transformacji,
- occupancy każdego stanu `>= 0.05`,
- maksymalna odległość powtórzenia pomiędzy startami `<= 1e-6`.

Komenda preflight (nie uruchamia rolling):

```bash
cd worker
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m regime_sentinel_worker.main msgarch-preflight \
  --rscript ../.runtime/msgarch-r4.4.3-msgarch2.51/env/bin/Rscript \
  --artifacts /tmp/ss-msgarch-preflight-run \
  --baseline-artifacts ../artifacts/baseline
```

Pełny etap `regime` uruchamia preflight, a rolling MS-GARCH dopiero wtedy, gdy bramka preflight dla obu instrumentów została zaliczona.

Jeśli preflight nie przejdzie lub rolling MS-GARCH nie osiąga stabilności, pipeline przechodzi do fallback (`fallback_not_ms_garch`), więc rolling OOS MS-GARCH nie uruchamia się automatycznie poza przypadkiem przejścia całej bramki.

```bash
cd worker
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
python3 -m regime_sentinel_worker.main regime \
  --rscript ../.runtime/msgarch-r4.4.3-msgarch2.51/env/bin/Rscript \
  --artifacts /tmp/ss-regime-run \
  --baseline-artifacts ../artifacts/baseline
```

## 8. Dostarczone wyniki

- `artifacts/baseline` — pełny wynik `GARCH(1,1)` dla BTC-USD i ETH-USD.
- `artifacts/msgarch-attempt-1` — kontrolowany preflight MSGARCH, bez rolling i bez fallback.
- `artifacts/msgarch-attempt-2` — kontrolowany preflight MSGARCH po poprawce inicjalizacji `nu`, z dalszym odrzuceniem bramki.
- `artifacts/fallback` — wynik `fallback_not_ms_garch` po niezaliczonej bramce.
- `artifacts/posthoc-diagnostics` — raport diagnostyczny opisowy, bez podejmowania modelowego zwycięzcy.
- `artifacts/provisional` — raport porównawczy bazowy i fallback (modele uwzględnione bez finalnej decyzji zwycięzcy).

## 9. Interpretacja

`fallback_not_ms_garch` **nie jest** MS-GARCH.

Brak rolling OOS MS-GARCH oznacza, że nie powstało rozstrzygnięcie porównawcze na bazie pełnego OOS dla tego wariantu; to nie jest dowodem przegranej MS-GARCH, tylko braku formalnie domkniętego toku porównawczego w `thesis-v1`.

## Sprawdzenie zgodności poleceń z kodem

Wszystkie polecenia komendowe w tej instrukcji pochodzą bezpośrednio z implementacji w:
`worker/src/regime_sentinel_worker/main.py`, `worker/src/regime_sentinel_worker/pipeline/*.py`, `worker/src/regime_sentinel_worker/regime.py`, `worker/r/msgarch_fit.R`, `worker/r/bootstrap_msgarch_env.sh` oraz istniejących testów w `worker/tests/*`.

# Contracts

Kontrakty są wersjonowane i domyślnie zamknięte na nieznane pola. Zmiana łamiąca kompatybilność wymaga
nowej wersji pliku i jawnej migracji konsumenta.

## Experiment protocol

`experiment-protocol.v1.schema.json` opisuje zamrożone decyzje metodologiczne eksperymentu. Instancją
produkcyjną v1 jest `experiments/thesis-v1/protocol.json`. JSON Schema kontroluje kształt danych, a
`regime_sentinel_worker.experiment_protocol` sprawdza inwarianty domenowe: hashe snapshotu, zakres OOS,
information boundary, konwencję VaR/ES i uczciwe użycie fallbacku.

Po wygenerowaniu pierwszego wyniku OOS zmiana wartości wpływającej na wynik wymaga nowego `protocolId`.

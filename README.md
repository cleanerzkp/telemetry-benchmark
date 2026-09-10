# Telemetry Benchmark | TB-01

Lokalne rozpoznanie metod detekcji anomalii: monitoring progowy, niezależne Isolation Forest dla kanałów oraz wielowymiarowy autoenkoder odszumiający PyTorch. Jeden zbiór SMD i jedna maszyna, machine-1-2. Zakres obejmuje implementację metod, kalibrację, walidację, zamrożony test i analizę incydentów.

## Wynik referencyjny

| Metoda | Precision | Recall | F1 | FA / dobę |
|---|---:|---:|---:|---:|
| Średnia ruchoma | 0.095 | 0.857 | 0.171 | 5.64 |
| Isolation Forest 1D | 0.118 | 0.286 | 0.167 | 1.54 |
| Autoenkoder DAE | 0.095 | 0.857 | 0.171 | 5.54 |

Holdout: 14 037 próbek, 7 incydentów. Baseline wybrany na walidacji: Isolation Forest 1D.
Autoenkoder nie wykazał przewagi F1 nad średnią ruchomą.

[Raport techniczny PDF](reports/RAPORT.pdf) · [Protokół](PROTOCOL.md) · [Pełny pakiet z danymi i modelem](https://github.com/cleanerzkp/telemetry-benchmark/releases/latest)

![Przebieg testowy](reports/benchmark.png)

## Artefakty

- `PROTOCOL.md`: metodyka ustalona przed treningiem.
- `config.json`: pojedyncza konfiguracja, bez strojenia.
- `data/manifest.json`: wersja źródła, indeksy podziału i SHA-256.
- `results/freeze.json`: model, progi i baseline wybrany na walidacji.
- `results/metrics.json`: wyniki trzech metod na końcowym holdoucie.
- `results/alerts.csv`, `results/incidents.csv`: audyt alarmów i dopasowań.
- `results/scores.npz`: zapis predykcji oraz błędów rekonstrukcji.
- `reports/RAPORT.pdf`: dwustronicowy raport techniczny.
- `reports/benchmark.png`: pełny przebieg testowy z detekcjami.

## Środowisko i interfejs

Python 3.14; PyTorch, NumPy, scikit-learn. Referencyjny przebieg wykonano lokalnie na CPU, bez usług zewnętrznych. Dokładne wersje: `requirements-lock.txt`.

```sh
python -m pip install -r requirements-lock.txt
python -m unittest discover -s tests
python run.py prepare
# Protokół, konfiguracja, kod i manifest muszą być zapisane w Git przed fit.
python run.py fit
# results/freeze.json musi być zapisany w Git przed test.
python run.py test
python render_report.py
```

W dostarczonym pakiecie dane są przygotowane, a przebieg zakończony. Polecenia prepare/fit/test celowo odmawiają nadpisania istniejących artefaktów. Raport można ponownie renderować z zapisanych wyników. Ponowny eksperyment wymaga osobnej kopii roboczej, jawnego identyfikatora i nowej historii; ponowne użycie tych samych danych nie staje się niezależnym testem.

## Pochodzenie danych

Repozytorium: https://github.com/NetManAIOps/OmniAnomaly
Commit: `7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4`.
Źródła: `ServerMachineDataset/train/machine-1-2.txt`, `test/machine-1-2.txt`, `test_label/machine-1-2.txt`.
Oryginalne bajty zapisano odpowiednio w `data/train.raw`, `data/evaluation.raw`, `data/labels.raw`. Licencja źródłowa: `data/SOURCE_LICENSE`.

Dane i model dołączono do pakietu lokalnego; nie są automatycznie dodawane do Git. Git zawiera manifest z hashami, kod, protokół i wyniki. Pełny pakiet ZIP z danymi i wytrenowanym modelem jest dostępny w sekcji Releases repozytorium GitHub. Sam klon Git zawiera kod, wyniki i raporty; do wykonania audit.py potrzebny jest także model z pakietu. Dane i model w wydaniu odpowiadają hashom w zamrożonych artefaktach.

## Granice interpretacji

SMD zapewnia 38 kanałów tej samej maszyny, a nie heterogeniczne źródła radiowe. Podział na ciągłe bloki chronologiczne nie jest podziałem na niezależne przebiegi testbedu. FA/dobę zakłada regularne próbkowanie minutowe wskazane w literaturze; pliki źródłowe nie zawierają timestampów.

Materiał dokumentuje wykonany eksperyment ML i jego metodykę. Nie zawiera deklaracji dorobku zawodowego ani walidacji rozwiązania radiokomunikacyjnego.

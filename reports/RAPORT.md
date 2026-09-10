# TB-01: Detekcja anomalii w telemetrii serwerowej

DAE uzyskał wyższe F1 niż baseline wybrany na walidacji: 0.171 wobec 0.167. Dopasował 6 z 7 incydentów przy 5.54 FA/dobę; baseline dopasował 2 z 7 przy 1.54 FA/dobę. Rolling uzyskał F1 0.171: nie wykazano przewagi DAE nad monitoringiem progowym. To obserwacja z jednego bloku, nie dowód przewagi ogólnej ani gotowości wdrożeniowej.

| Metoda | Precision | Recall | F1 | FA/dobę |
|---|---:|---:|---:|---:|
| Średnia ruchoma | 0.095 | 0.857 | 0.171 | 5.64 |
| Isolation Forest 1D | 0.118 | 0.286 | 0.167 | 1.54 |
| Autoenkoder DAE | 0.095 | 0.857 | 0.171 | 5.54 |

Średnia ruchoma: 6/7 trafień, 63 epizodów, 55 FA, 2 duplikatów; Isolation Forest 1D: 2/7 trafień, 17 epizodów, 15 FA, 0 duplikatów; Autoenkoder DAE: 6/7 trafień, 63 epizodów, 54 FA, 3 duplikatów.

![Pełny holdout](benchmark.png)

Publiczny zbiór nie odwzorowuje heterogenicznej telemetrii radiowej. Brak kontrolowanych scenariuszy degradacyjnych. Rozpoznanie służy weryfikacji metodyki i narzędzi, nie walidacji rozwiązania. Jedna maszyna, jeden seed i niewielka liczba incydentów nie pozwalają na wnioski o generalizacji. Ciągłe bloki SMD nie są niezależnymi przebiegami testbedu. Nie badano few-shot, grafów ani diagnostyki przyczynowej. Wyniki bazowe nie rozstrzygają problemu badawczego planowanego projektu.

Pełna metodyka: ../PROTOCOL.md. Szczegóły i źródła: RAPORT.pdf.

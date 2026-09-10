# Protokół TB-01: porównanie trzech detektorów anomalii telemetrycznych

Status: ustalony przed treningiem i odczytem wyników machine-1-2. Parametry są w config.json. Nie prowadzimy strojenia, przeszukiwania ani selekcji seedów. Cel: sprawdzenie metodyki i narzędzi rozpoznania wstępnego.

## Dane i zakres

Jeden zbiór: SMD, jedna maszyna: machine-1-2, 38 anonimowych kanałów. Źródło: NetManAIOps/OmniAnomaly, commit 7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4.
Wcześniejszy eksperyment dotyczył machine-1-1; jego wyniki są znane i nie stanowią testu tej wersji. Machine-1-2 wybrano jako następną w numeracji, przed oglądaniem jej metryk. Pilotaż TB-01 nie jest porównywalny liczbowo z poprzednią wersją.

Oficjalny plik train: pierwsze 80% na trening, przerwa 120 próbek, reszta na kalibrację progów. Oficjalny plik test: pierwsze 40% na walidację wyboru baseline'u, przerwa 120 próbek, reszta na końcowy holdout. Zaokrąglenie granic w dół. Dokładne indeksy i SHA-256 zapisuje data/manifest.json.

Całe ciągłe bloki trafiają do jednej roli. Nie losujemy okien między zbiorami, nie przenosimy kontekstu między blokami. SMD nie zawiera identyfikatorów niezależnych przebiegów eksperymentalnych: NIE twierdzimy, że ten podział jest walidacją na całych niezależnych przebiegach testbedu. To jawne ograniczenie dostępnego zbioru.

W każdym bloku pierwsze 60 próbek stanowi kontekst i jest pominięte w ocenie wszystkich metod. Każdy incydent jest maksymalnym ciągiem etykiet 1 w ocenianym bloku. Incydenty przecięte granicą są liczone jako obserwowane fragmenty i oznaczane w metrykach. Podziału nie przesuwamy po odczycie etykiet. Długości plików i ich hashe wolno sprawdzić wcześniej; wartości holdoutu i jego etykiety są parsowane dopiero w fazie test.

Przyjmujemy interwał 1 min zgodnie z Li i in., „An Empirical Analysis of Anomaly Detection Methods for Multivariate Time Series”, tabela I: https://netman.aiops.org/wp-content/uploads/2024/08/Empirical_Analysis.pdf . CSV nie ma timestampów; przeliczenie na dobę zakłada regularność i brak luk. Nie korzystamy z danych ani zmienionych etykiet tej publikacji.

## Trzy metody

1. Rolling: max po kanałach z |x(t) - średnia(x(t-60:t))| / skala_treningowa. Średnia jest przyczynowa i wyklucza bieżącą próbkę. Model reprezentuje monitoring progowy.
2. IForest-1D: osobny Isolation Forest dla każdego kanału, z domyślnymi parametrami scikit-learn, seed 42, n_jobs=1. Wynik = max z -score_samples po 38 kanałach. Jest to jedna metoda z 38 niezależnymi modelami jednozmiennowymi; nie uczy relacji między kanałami. Wyniki kanałów nie są prawdopodobieństwami.
3. DAE: odszumiający autoenkoder PyTorch, 38-32-8-32-38, ReLU w warstwach ukrytych, wyjście liniowe. Losowe maskowanie 15%, 30 epok, batch 256, Adam z domyślnymi parametrami, MSE. Seed 42, CPU, obliczenia deterministyczne. Wynik = średni kwadrat błędu odtworzenia wszystkich kanałów bez maskowania w inferencji.

Średnia i std pochodzą wyłącznie z treningu. Skala = max(std, 0.01) dla stabilności prawie stałych kanałów; nie obcinamy wartości testowych. Architektura i parametry własnych metod nie mają uniwersalnych „ustawień domyślnych”: powyższe są pojedynczą konfiguracją ustaloną z góry. Brak early stopping i strojenia na walidacji. Tasowanie batchy odbywa się wyłącznie wewnątrz treningu, po podziale danych.

## Kalibracja i wybór baseline'u

Dla każdej z trzech metod próg to 99. percentyl jej score na nieetykietowanej kalibracji; alarm gdy score > próg. Kalibracja nie zakłada dostępu do etykiet incydentów i nie gwarantuje 1% fałszywych alarmów poza tym blokiem.

Spośród Rolling oraz IForest-1D wybieramy baseline o najwyższym F1 incydentowym na walidacji. Remis: mniej fałszywych alarmów/dobę, następnie porządek nazwy metody. Progi i modele nie są zmieniane podczas wyboru. Wszystkie trzy metody trafią do tabeli testowej; porównaniem głównym jest DAE vs baseline wybrany na walidacji, nawet gdy drugi baseline wygra test.

## Agregacja i metryki

Sąsiednie surowe alerty z przerwą do 30 próbek tworzą jeden epizod alarmowy. Reguła nie używa etykiet; epizod przechowuje rzeczywiste przedziały przekroczeń. Przerwy między alertami NIE stają się dodatnimi predykcjami. Dopasowanie incydentu wymaga rzeczywistego przekroczenia progu w jego obrębie.

Dopasowanie chronologiczne jeden-do-jednego: każdy epizod alarmowy dopasowujemy do najwcześniejszego jeszcze niedopasowanego incydentu z rzeczywistym przekroczeniem. Jeden długi alarm nie może dostać wielu trafień. Kolejne niedopasowane alarmy przecinające wykryty incydent są duplikatami.

Precision = liczba dopasowań / liczba epizodów alarmowych. Recall = liczba dopasowań / liczba incydentów. F1 = średnia harmoniczna P i R. Brak alarmów lub brak incydentów: odpowiednia metryka = 0, a surowe liczności zawsze są raportowane. Duplikaty obniżają precision.

Fałszywy alarm = epizod bez rzeczywistego przekroczenia w żadnym incydencie. FA/dobę = liczba takich epizodów / (liczba ocenianych próbek / 1440). Duplikaty i liczba niedopasowanych epizodów są dodatkowo raportowane osobno. Raport obejmuje także odsetek surowych dodatnich próbek i medianę opóźnienia trafień, by szeroki alarm nie ukrywał słabej użyteczności.

Nie stosujemy point adjustment. Nie stroimy długości agregacji. Nie liczymy testowych „best F1”.

## Kolejność i artefakty

A. Commit PROTOCOL.md, konfiguracji, kodu i manifestu, przed treningiem.
B. Trening, kalibracja, wybór baseline'u na walidacji. Zapis i osobny commit results/freeze.json zawierającego hashe kodu, danych, modelu, progi i wyniki walidacji.
C. Polecenie test tworzy nieusuwany znacznik TEST_OPENED.json przed odczytem holdoutu. Tylko jeden przebieg; powtórne polecenie odmawia. Kod/protokół muszą zgadzać się z zamrożonymi hashami.
D. Zapis surowych score, etykiet, CSV alarmów/incydentów, tabeli i raportu. Osobny commit z wynikami. Rendering raportu może być powtarzany z zapisanych wyników; nie uruchamia modeli ani nowej oceny.

Wykres: pełny holdout, kanał o największej wariancji treningowej, etykiety incydentów i surowe alarmy każdej metody. Kanału ani zakresu nie wybieramy pod korzystny wynik. Historia Git jest lokalnym zapisem kolejności, nie niezależnym znacznikiem czasu.

## Ograniczenia wnioskowania

Publiczny zbiór nie odwzorowuje heterogenicznej telemetrii radiowej. Brak kontrolowanych scenariuszy degradacyjnych. Rozpoznanie służy weryfikacji metodyki i narzędzi, nie walidacji rozwiązania. Jedna maszyna i seed nie pozwalają uogólniać skuteczności. Nie realizujemy few-shot, grafów, diagnostyki przyczynowej ani porównania z SOTA.

Ewentualny dobry wynik bazowego autoenkodera na SMD nie rozstrzyga problemu badawczego planowanego projektu. Ewentualny słaby wynik pozostaje w raporcie bez zmiany protokołu po teście.

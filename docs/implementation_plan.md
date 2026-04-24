# Plan Refaktoryzacji: Inteligentne i Dynamiczne Rozpoznawanie BOFF

Rozumiem już dokładnie geometrię BOFFów w STO. Główne założenia, o których mówiliśmy:
* Prawdziwa struktura to **5 osobnych foteli (seatów)** rozmieszczonych w dwóch kolumnach (3 po lewej, 2 po prawej). Każdy fotel to poziomy wiersz mieszczący do 4 umiejętności.
* Fotel może być "mieszany" (np. główna profesja + specjalizacja), więc wymuszanie jednej profesji dla całego fotela metodą głosowania psuje wyniki.
* Detekcja nazwy statku w `SPACE_MIXED` wymaga usprawnienia, aby można było rzetelnie wykorzystać bazę `ship_list.json`.
* Kolory ramek dają wysoką pewność bazowej profesji, ale trzeba uwzględnić specjalizacje (które mogą z nim współdzielić fotel).

Oto zaktualizowany plan podzielony na moduły, w których dokonamy zmian:

## 1. Wyodrębnienie geometryczne (Geometria Foteli)
Zmieniamy sposób, w jaki `LayoutDetector` raportuje zgrupowane ikony. Zamiast z góry zgadywać i wrzucać bboxy do "Boff Tactical" itp., detektor będzie raportował pozycje jako logiczne jednostki:
* Zwracamy układ: `Seat_L1`, `Seat_L2`, `Seat_L3` (lewa kolumna) oraz `Seat_R1`, `Seat_R2` (prawa kolumna).
* Każdy "Seat" zawiera do 4 bbxów. Przestajemy polegać na "większościowym" kolorze zmuszając cały rząd do jednej szufladki. 
> [!NOTE]
> Dzięki temu wiemy, które konkretnie umiejętności ze sobą sąsiadują i należą do tego samego oficera.

## 2. Nowa logika dobierania kandydatów (Węższe wyszukiwanie)
W module `warp_importer.py`, przetwarzając wykryte fotele, zastosujemy elastyczną strategię "konsensusu kolorystycznego + specjalizacje":
1. Określamy wiodący kolor dla fotela (Tac/Eng/Sci) używając `_classify_boff_profession()`.
2. Zamiast szukać w pełnej bazie BOFFów (ponad 600 skilli), instruujemy `IconMatcher`, aby szukał **tylko w umiejętnościach wybranej profesji ORAZ umiejętnościach specjalizacyjnych** (Miracle Worker, Pilot, Intel, Command, Temporal).
3. Jeśli żadna ikona w tym węższym zbiorze nie przekroczy pewnego wyższego progu ufności, dopiero wtedy "otwieramy" pulę na pozostałe profesje.
> [!TIP]
> Zmniejszenie słownika poszukiwań z 600+ na ok. 100-150 ikonek drastycznie zredukuje pomyłki, na które natrafiliśmy w testach.

## 3. Propagacja Feedbacku w WARP CORE (Trenerze)
Zmodyfikujemy `AnnotationWidget`, aby ułatwić pracę człowieka i uczyć model "w locie":
* Kiedy klikniesz i zatwierdzisz umiejętność (np. przypiszesz ją jako "Cannon: Rapid Fire"), system odczyta jej profesję (Tactical).
* Natychmiast "zaraża" **tylko ten jeden fotel** – wymuszając zawężony słownik (Tactical + Specializations) dla pozostałych (jeszcze niepotwierdzonych) ikonek *w tym samym rzędzie i kolumnie*.
* Następnie system automatycznie, ponownie uruchamia rozpoznawanie (`match()`) dla tych sąsiadujących ikonek. To daje magiczny efekt: poprawienie jednego skilla naprawia resztę oficera.

## 4. Poprawa OCR Statków i Cross-Check (Krok Równoległy)
Aby wykorzystać informacje o układzie foteli statku (z `ship_list.json`), musimy mieć wyższą skuteczność odczytu OCR jego typu:
* Zoptymalizujemy paski skanowania `TextExtractor`, być może dynamicznie lokalizując tekst "Tier" i od niego odpowiednio mierząc obszar zawierający Typ statku.
* Po dobrym rozpoznaniu statku, wynik `ship_list.json` będzie działał jako finalny walidator – dowiemy się, że na tym statku w ogóle nie ma np. fotela Science, więc algorytm wykluczy go jako odpowiedź.

## Wymagane Zmiany (Które pliki ulegną modyfikacji)
#### [MODIFY] `warp/recognition/layout_detector.py`
Usunięcie sztywnego `Counter(profs).most_common(1)` i grupowania, zastąpienie zwrotem struktury np. `Seat_X_Y`.

#### [MODIFY] `warp/warp_importer.py`
Adaptacja logiki na czytanie struktury foteli. Dodanie nowej procedury `_build_slot_candidates`, która elastycznie łączy wykryty kolor bazowy i klasy Specjalizacji.

#### [MODIFY] `warp/trainer/annotation_widget.py`
Nasłuchiwanie na zatwierdzenie bboxa, identyfikacja fotela (po współrzędnej `y`) i wymuszenie cichego `_run_matcher()` z nowymi, zacieśnionymi `valid_types` dla sąsiadów.

#### [MODIFY] `warp/recognition/text_extractor.py`
Tuning okna OCR na nazwę/typ statku na podstawie analizy błędów z naszego skryptu testowego.

## Open Questions
1. **Nazewnictwo Slotów dla SETS**: Aplikacja SETS (`src/sets.py`) oczekuje starych nazw slotów do zapisu konfiguracji (np. `'Boff Tactical'`). Zgaduję, że `warp_importer` na samym końcu, kiedy już dowiemy się czym fizycznie jest dany skill (bo zwrócił to IconMatcher), będzie po prostu wsadzał go do ogólnego worka z tą nazwą profesji (czyli np. jak IconMatcher wykryje "Override Subsystem Safeties", to dopiszemy go do ogólnego slotu `'Boff Intelligence'` dla SETS, ignorując w jakim fizycznym "Seat" siedział)?
2. **Kolejność Prac**: Sugeruję najpierw zaimplementować nową strukturę detekcji "Seatów" w układzie i zmienić słowniki w `IconMatcherze` (punkty 1 i 2). Powinniśmy od razu widzieć wzrost accuracy w skrypcie `test_icon_matcher.py`. Następnie propagację w trenerze, a na koniec OCR statków. Czy zgadzasz się z takim podziałem?

>Nazewnictwo Slotów dla SETS:

Trochę nie rozumiem pytania. Na Twoim przykładzie: IconMatcher wykryje "Override Subsystem Safeties", to dopiszemy go do ogólnego slotu 'Boff Intelligence' dla SETS. 
no jeśli wykrył OSS, no to jest to Boff int, nie do końca rozumiem co masz na myśli: w jakim fizycznym "Seat" siedział. 
Chodzi ci o ułożenie w SETS? Że gdzie w którym seat powinien być być ten skill?
Jeśli tak, to jest bardziej skomplikowane. W SETS kolejność i ułozenie boffów nie jest taka sama jak w grze. W SETS Jest to posortowana lista setaów danego typu statku, od największej ilości skili, do najmniejszej. Możesz sprawdzić kod i potwierdzić.
Czyli w SETS np. na statku Fleet Yamaguchi Support Cruiser masz od góry: Engineering/Temporal - 4 sloty, Universal - 3 sloty, Science - 3 sloty, Tactical - 2 sloty, Tactical - 1 slot.
Więc rozpoznane seaty musisz poukładać, by pasowały do siatki statku. Ale to chyba na razie dalszy temat? Na teraz skupiamy się na samym wykrywaniu w WARP i WARP CORE raczej.
>Kolejność Prac: Sugeruję najpierw zaimplementować nową strukturę detekcji "Seatów" w układzie i zmienić słowniki w IconMatcherze (punkty 1 i 2). Powinniśmy od razu widzieć wzrost accuracy w skrypcie test_icon_matcher.py. Następnie propagację w trenerze, a na koniec OCR statków.

OCR statków właściwie już jest. więc pewnie detakcja seatów, nie do końca wiem co za słowniki chcesz zmieniać, z czego na co.

## Tasks:
- [ ] Zmiana struktury w `layout_detector.py` tak, aby zwracała "Seaty" (fizyczne stanowiska), a nie narzucała z góry pełną profesję (Boff Tactical).
- [ ] Modyfikacja `warp_importer.py` w funkcji `_build_slot_candidates`, aby przyjmowała stanowiska z `LayoutDetector`, klasyfikowała wiodący kolor z użyciem `_classify_boff_profession` i ograniczała kandydatów IconMatchera do [Wiodąca Profesja + Specjalizacje].
- [ ] Poprawka w `warp_importer.py` w trakcie iteracji po wykrytych bboxach: po rozpoznaniu dokładnej nazwy skilla (np. "Cannon: Rapid Fire") odczytanie jego profesji ze słownika i przemapowanie `slot_name` z nazwy geometrycznej (np. `Boff Seat 1`) na nazwę slotu w SETS (np. `Boff Tactical`).
- [ ] Implementacja propagacji decyzji użytkownika w oknie Trenera (`AnnotationWidget`): gdy użytkownik zatwierdzi (potwierdzi) jedną umiejętność w fotelu, system w tle uaktualnia ograniczenia kandydatów dla pozostałych umiejętności w tym fotelu i puszcza na nowo IconMatcher, automatycznie podnosząc skuteczność.

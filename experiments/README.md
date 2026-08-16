# Experimenty

Osm reportovacích notebooků, které dokládají každé rozhodnutí v návrhu pipeline
a reprodukují všechny tabulky a grafy z praktické části bakalářské práce.

## Notebooky

| # | Notebook | Otázka | Kapitola práce |
|---|---|---|---|
| 01 | [`01_analyza_dat`](notebooks/01_analyza_dat.ipynb) | Jak vypadají faktury, které do KEM reálně přicházejí? | 3 (obr. 3.1–3.5) |
| 02 | [`02_predzpracovani_obrazu`](notebooks/02_predzpracovani_obrazu.ipynb) | Pomáhá agresivnější předzpracování obrazu, nebo škodí? | 4.2 (tab. 4.1) |
| 03 | [`03_prostorove_kodovani`](notebooks/03_prostorove_kodovani.ipynb) | Jak předat modelu strukturu tabulek? | 4.2 (obr. 4.2) |
| 04 | [`04_jazyk_promptu`](notebooks/04_jazyk_promptu.ipynb) | Záleží na jazyce zadání? | 4.2 (obr. 4.3) |
| 05 | [`05_strategie_a_modely`](notebooks/05_strategie_a_modely.ipynb) | Jak formulovat instrukci a který model zvolit? | 4.2 (obr. 4.4–4.8, tab. 4.2) |
| 06 | [`06_multimodalni_pristup`](notebooks/06_multimodalni_pristup.ipynb) | Může Vision-LLM nahradit OCR? | 4.2–4.3 (obr. 4.10, 4.11) |
| 07 | [`07_evaluace_ds1`](notebooks/07_evaluace_ds1.ipynb) | Jak si řešení vede na 573 reálných fakturách? | 5.1 (tab. 5.1–5.2, obr. 5.1–5.6) |
| 08 | [`08_generalizace_ds3`](notebooks/08_generalizace_ds3.ipynb) | Generalizuje na neznámé dodavatele? | 5.2–5.5 (tab. 5.3, obr. 5.7) |

Notebooky na sebe navazují: každý fixuje jedno rozhodnutí, které vstupuje do
následujícího. Notebooky `02`–`06` pracují výhradně s vývojovou sadou **DS2**,
aby zůstala sada **DS1** nedotčená pro závěrečné vyhodnocení v notebooku `07`.

```
01 analýza dat        →  jaké dokumenty systém potká
02 předzpracování     →  P4: stupně šedi + potlačení šumu
03 prostorové kódování→  JSON-light
04 jazyk promptu      →  cs_en (instrukce anglicky, pole česky)
05 strategie + model  →  few-shot + GPT-4.1          ── hlavní cesta
06 multimodální model →  gpt-4.1-mini + RAG          ── záložní cesta
07 evaluace DS1       →  F1 = 0,85 · 7 z 9 kritérií
08 generalizace DS3   →  F1 = 0,94 · 6 z 8 kritérií
```

## Spuštění

Notebooky jsou uloženy **včetně výstupů** — lze je jen prohlédnout, bez čehokoli
instalovat. Pro vlastní spuštění stačí základní vědecký stack; API klíč,
Tesseract ani samotné faktury nejsou potřeba:

```bash
pip install -e ".[notebooks]"
jupyter lab experiments/notebooks
```

## Odkud berou data

Každý notebook čte předpočítané výsledky z [`experiments/data/`](data/).
Naměřené hodnoty se tak nemění mezi spuštěními a vždy odpovídají číslům v práci —
opakované volání jazykového modelu by kvůli jeho nedeterminismu dávalo pokaždé
mírně jiná čísla.

| Složka | Obsah | Vznikla z |
|---|---|---|
| `01_analyza_dat/` | agregovaná statistika korpusu KEM | `scripts/build_corpus_stats.py` |
| `02_predzpracovani/` | 1 620 měření OCR (180 faktur × 9 konfigurací) | běh experimentu na DS2 |
| `03_prostorove_kodovani/` | 456 měření (152 faktur × 3 formáty) | běh experimentu na DS2 |
| `04_jazyk_promptu/` | 1 620 měření (3 jazyky × strategie) | běh experimentu na DS2 |
| `05_strategie_modely/` | Fáze 1 (4 strategie) a Fáze 2 (7 modelů) | běh experimentu na DS2 |
| `06_multimodalni/` | 24 konfigurací Vision-LLM + přínos RAG | běh experimentu na DS2 |
| `07_evaluace_ds1/` | 573 faktur — výsledek na fakturu i na pole | `scripts/run_ds1_final.py` |
| `08_generalizace_ds3/` | 162 syntetických faktur | `scripts/run_ds3_final.py` |

### Ochrana osobních údajů

Reálné faktury obsahují osobní údaje a v repozitáři nejsou. Tabulky výsledků
proto **neobsahují žádnou hodnotu z faktury** ani název souboru — jen komoditu,
třídu kvality, latenci, náklady a stav každého pole (`tp` / `fp` / `fn`).
Metriky se počítají z těchto stavů, takže odstraněním hodnot se žádný výsledek
uvedený v práci nemění.

Anonymizaci provádí [`scripts/build_thesis_datasets.py`](../scripts/build_thesis_datasets.py),
který se spouští nad surovými výstupy běhu na lokálním stroji.

## Jednotný styl grafů

Modul [`thesis_style.py`](thesis_style.py) drží stejnou paletu i typografii jako
obrázky v práci, aby grafy z notebooků a grafy v textu tvořily jeden celek.

```python
import sys; sys.path.insert(0, "..")
from thesis_style import *
pouzij_styl()
```

Komodity mají pevně přiřazenou barvu napříč všemi notebooky (paleta ColorBrewer
Set2), u srovnání metod zvýrazňuje tmavě modrá zvolenou variantu a červená
protikandidáta. Protože je paleta pastelová a samotný odstín by nestačil jako
jediný nositel informace, grafy identitu vždy kódují ještě popiskem osy nebo
přímou hodnotou u sloupce.

# Automatická extrakce dat ze skenovaných faktur dodavatelů energií

Nástroj, který ze skenované faktury za energie vytěží strukturovaná data
a nahradí tak jejich ruční přepis do systému **Krajského energetického
managementu (KEM)** provozovaného **Datovým centrem Ústeckého kraje**.

Praktická část bakalářské práce · Kseniia Mahalias · Přírodovědecká fakulta UJEP, 2026

---

## Proč

Faktury za energie zpracovávají v KEM převážně účetní bez hlubších technických
znalostí v energetice. Ruční přepis jedné faktury trvá v průměru **6,4 minuty**
a při opisování složitých technických údajů vznikají překlepy a záměny polí.

Uživatelé navíc pod tlakem času vyplňují jen povinná pole — analýza databáze
ukázala, že **nepovinné technické údaje obsahuje jen 18 % záznamů**, přestože se
na faktuře fyzicky nacházejí. Systém tak přichází o víc než 80 % detailních
datových bodů využitelných pro energetickou analytiku.

Řešení vytěží **všechna pole bez ohledu na jejich „povinnost“** za 18 sekund
a 0,71 Kč na dokument.

## Výsledky

Měřeno na **573 reálných fakturách** od 19 dodavatelů, které systém neviděl
v žádné fázi vývoje.

| Ukazatel | Výsledek | Požadavek |
|---|---|---|
| F1 skóre (průměr přes komodity) | **0,85** | ≥ 0,60 ✅ |
| Přesnost čísla faktury | **94,4 %** | ≥ 85 % ✅ |
| Přesnost IČO dodavatele | **96,9 %** | ≥ 85 % ✅ |
| Přesnost IČO odběratele | **93,8 %** | ≥ 85 % ✅ |
| Podíl validních JSON výstupů | **100 %** | ≥ 98 % ✅ |
| Průměrná doba zpracování | **18,4 s** | ≤ 60 s ✅ |
| F1 na nejhorších skenech (Q3) | **0,774** | ≥ 0,45 ✅ |
| Přesnost částky s DPH | 57,5 % | ≥ 85 % ❌ |
| Přesnost data splatnosti | 71,0 % | ≥ 85 % ❌ |

**Splněno 7 z 9 akceptačních kritérií.** Obě nesplněná mají identifikovanou
příčinu i směr opravy — rozbor je v notebooku
[`07_evaluace_ds1`](experiments/notebooks/07_evaluace_ds1.ipynb).

Na syntetické sadě 162 faktur od **dodavatelů, které systém nikdy neviděl**,
dosahuje F1 **0,938** — řešení tedy nestojí na znalosti konkrétních šablon
([`08_generalizace_ds3`](experiments/notebooks/08_generalizace_ds3.ipynb)).

## Jak to funguje

Žádný jednotlivý přístup nefunguje nejlépe za všech okolností: textová cesta přes
OCR je přesnější a levnější u čitelných skenů, multimodální model pomůže tam, kde
OCR selže. Řešením je **kaskáda**, která mezi nimi podle potřeby přepíná.

```text
                      ┌──────────────────────────────────────────┐
   PDF  ─────────────►│  HLAVNÍ CESTA          85 % dokumentů    │
                      │                                          │
                      │  předzpracování   stupně šedi + odšumění │
                      │  OCR              Tesseract → JSON-light │
                      │  extrakce         GPT-4.1, few-shot      │
                      └────────────────────┬─────────────────────┘
                                           ▼
                                ┌──────────────────────┐
                                │  validace + kontrola │
                                │  čitelný JSON?       │
                                │  klíčová pole?       │
                                │  jistota modelu?     │
                                └──────┬────────┬──────┘
                                 ANO   │        │   NE
                                       │        ▼
                                       │  ┌──────────────────────────────────┐
                                       │  │  ZÁLOŽNÍ CESTA   15 % dokumentů  │
                                       │  │                                  │
                                       │  │  Vision-LLM   gpt-4.1-mini       │
                                       │  │  zero-shot + RAG (dodavatelé)    │
                                       │  └──────────────┬───────────────────┘
                                       ▼                 ▼
                                  ┌─────────────────────────┐
                                  │  strukturovaný JSON     │
                                  │  + confidence_score     │
                                  └─────────────────────────┘
```

Každý článek řetězu vzešel z měření, ne z odhadu — viz
[přehled experimentů](experiments/README.md):

| Rozhodnutí | Zvoleno | Proč |
|---|---|---|
| Předzpracování obrazu | stupně šedi + potlačení šumu | nejnižší chybovost z 9 konfigurací; agresivnější úpravy text ničí |
| Formát vstupu pro model | JSON-light | shodná přesnost jako HTML, výrazně méně tokenů |
| Jazyk zadání | instrukce anglicky, pole česky | angličtina pro uvažování, čeština bez rizika překladu |
| Promptovací strategie | few-shot | +9 p. b. přesnosti; řetězec úvah zvyšuje podíl nevyplněných polí na 34 % |
| Hlavní model | GPT-4.1 | nejlepší poměr přesnosti a ceny ze 7 testovaných |
| Záložní model | gpt-4.1-mini (Vision) | nejpřesnější **i** nejlevnější z multimodálních |
| Název dodavatele | RAG nad databází KEM | přesnost pole vzrostla z 0,40 na 1,00 |

Výstupem je JSON s **72 datovými poli** napříč šesti komoditami — elektřina NN
a VN, plyn maloodběr a velkoodběr, teplo a voda — doplněný o `confidence_score`,
podle kterého může KEM označit nejistá pole ke kontrole člověkem. Systém je
navržen jako nástroj pro **předvyplnění dat, nikoli jako plně autonomní řešení**.

## Rychlý start

### Docker

Nejrychlejší cesta — obraz obsahuje i Tesseract s českým jazykovým balíkem, který
se instaluje nejhůř.

```bash
docker compose up notebooks     # JupyterLab s experimenty → http://localhost:8888
docker compose up api           # REST API                 → http://localhost:8000/docs
```

### Lokálně

Pro pouhé prohlédnutí a spuštění experimentů stačí vědecký stack — API klíč,
Tesseract ani samotné faktury nejsou potřeba:

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
# source .venv/bin/activate                        # Linux / macOS

pip install -e ".[notebooks]"
jupyter lab experiments/notebooks
```

Pro spuštění celé pipeline nad vlastními fakturami:

```bash
pip install -e ".[pipeline,llm]"
cp .env.example .env            # doplňte OPENAI_API_KEY
```

> **Bez klíče `OPENAI_API_KEY` běží pipeline v degradovaném režimu.** Místo
> jazykového modelu se použije záložní extrakce regulárními výrazy — ta slouží
> jako srovnávací základ, ne jako produkční řešení. Vrací mnohem nižší jistotu,
> často `UNKNOWN` místo čísla faktury a zaměňuje odběratele s dodavatelem.
> Systém na to upozorní v logu i ve výstupním poli `warnings`; výsledky z tohoto
> režimu nelze importovat bez kontroly.

Cestu k Tesseractu si systém zjistí sám — na Linuxu, macOS i v kontejneru
z `PATH`, na Windows z obvyklého umístění instalátoru. Vynutit ji lze proměnnou
`TESSERACT_CMD`.

Navíc je potřeba **Tesseract 5 s českým jazykovým balíkem**:

```bash
sudo apt install tesseract-ocr tesseract-ocr-ces        # Debian / Ubuntu
# Windows: instalátor UB-Mannheim, při instalaci zaškrtnout „Czech“
```

## Experimenty

Osm reportovacích notebooků dokládá každé rozhodnutí v návrhu pipeline
a reprodukuje všechny tabulky a grafy z práce.

| # | Notebook | Otázka |
|---|---|---|
| 01 | [Analýza dat](experiments/notebooks/01_analyza_dat.ipynb) | Jak vypadají faktury, které do KEM reálně přicházejí? |
| 02 | [Předzpracování obrazu](experiments/notebooks/02_predzpracovani_obrazu.ipynb) | Pomáhá agresivnější předzpracování, nebo škodí? |
| 03 | [Prostorové kódování](experiments/notebooks/03_prostorove_kodovani.ipynb) | Jak předat modelu strukturu tabulek? |
| 04 | [Jazyk promptu](experiments/notebooks/04_jazyk_promptu.ipynb) | Záleží na jazyce zadání? |
| 05 | [Strategie a modely](experiments/notebooks/05_strategie_a_modely.ipynb) | Jak formulovat instrukci a který model zvolit? |
| 06 | [Multimodální přístup](experiments/notebooks/06_multimodalni_pristup.ipynb) | Může Vision-LLM nahradit OCR? |
| 07 | [Evaluace na DS1](experiments/notebooks/07_evaluace_ds1.ipynb) | Jak si řešení vede na 573 reálných fakturách? |
| 08 | [Generalizace na DS3](experiments/notebooks/08_generalizace_ds3.ipynb) | Generalizuje na neznámé dodavatele? |

Notebooky jsou uloženy **včetně výstupů** — lze je prohlédnout přímo v prohlížeči,
bez čehokoli instalovat. Podrobnosti v [`experiments/README.md`](experiments/README.md).

## Struktura repozitáře

```text
├── src/                    extrakční logika a REST API
│   ├── core/               OCR, extrakce, klasifikace, komodity, metriky, RAG
│   ├── domain/             datové entity (Pydantic)
│   ├── infrastructure/     databáze, klienti jazykových modelů
│   └── api/                FastAPI — nahrání dokumentu, výsledky
│
├── experiments/
│   ├── notebooks/          8 reportovacích notebooků
│   ├── data/               předpočítané výsledky měření (anonymizované)
│   └── thesis_style.py     jednotný styl grafů
│
├── scripts/
│   ├── run_ds1_final.py             hromadné zpracování evaluační sady
│   ├── run_ds3_final.py             totéž pro syntetickou sadu
│   ├── generate_synthetic_invoices.py   generátor sady DS3
│   ├── build_thesis_datasets.py     anonymizace výsledků pro notebooky
│   ├── build_corpus_stats.py        agregovaná statistika korpusu
│   └── rag_supplier_postprocess.py  dohledání dodavatele metodou RAG
│
├── data/                   syntetická sada DS3 (reálné sady mimo repozitář)
├── tests/                  testy (pytest)
└── docker/                 vstupní bod kontejneru
```

## Data a ochrana osobních údajů

Reálné faktury sad DS1 a DS2 **v repozitáři nejsou**, a to ani jejich ground
truth — ten obsahuje IČO odběratelů i dodavatelů, kódy odběrných míst (EAN, EIC)
a fakturované částky. Vše je uloženo na Google Disku, přístup byl poskytnut
vedoucímu a oponentovi práce.

Zveřejněné tabulky výsledků **neobsahují žádnou hodnotu z faktury** ani název
souboru; jen komoditu, třídu kvality, latenci, náklady a stav každého pole
(`tp` / `fp` / `fn`). Metriky se počítají z těchto stavů, takže anonymizací se
žádný výsledek uvedený v práci nemění. Podrobnosti v [`data/README.md`](data/README.md).

Syntetická sada **DS3 je v repozitáři kompletní** včetně PDF — všech 18 dodavatelů
je fiktivních.

## Reprodukce výsledků

Notebooky čtou předpočítané výsledky, takže naměřené hodnoty se mezi spuštěními
nemění a vždy odpovídají číslům v práci. Opakované volání jazykového modelu by
kvůli jeho nedeterminismu dávalo pokaždé mírně jiná čísla.

Kdo má přístup k reálným datům, může celý běh zopakovat:

```bash
python scripts/run_ds1_final.py          # plný běh nad DS1 (573 faktur)
python scripts/run_ds1_final.py --limit 20   # zkušební běh
python scripts/build_thesis_datasets.py  # přepočet podkladů pro notebooky
```

## Kvalita kódu

```bash
pip install -e ".[dev]"
ruff check src/ tests/ scripts/     # lint
pytest                              # 76 testů
```

Obojí hlídá i [CI](.github/workflows/ci.yml) při každém pushi.

## Limity

* **Řídce zastoupená pole** — u sazeb za rezervovanou kapacitu nebo distribuci má
  systém k dispozici málo příkladů a přesnost silně kolísá.
* **Závislost metody RAG na referenční databázi** — chybí-li dodavatel v databázi
  KEM, přiřazení selže a dokument zbytečně postupuje na záložní cestu. Produkční
  nasazení vyžaduje proces průběžné aktualizace seznamu dodavatelů.
* **Závislost na externím API** — řešení stojí na OpenAI API třetí strany: riziko
  změny cen, výpadku služby i přenosu dat mimo infrastrukturu DCÚK.
* **Aktuálnost modelů** — experimenty proběhly s modely řady GPT-4; s příchodem
  novějších generací je vhodné srovnání zopakovat.
* **Neověřené dotrénování** — z metod doménové adaptace byla ověřena pouze RAG.

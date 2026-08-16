# Data

## Co v repozitáři je

| Cesta | Obsah |
|---|---|
| `ds3/` | **kompletní syntetická sada** — 162 PDF i ground truth |

Nic víc — a je to záměr. Sada DS3 je celá vygenerovaná, všech 18 dodavatelů
i tři odběratelé jsou fiktivní, takže ji lze zveřejnit bez výhrad.

## Co v repozitáři není

**Faktury sad DS1 a DS2, jejich ground truth a referenční korpus KEM.**
Jde o reálné dokumenty; ground truth k nim navíc obsahuje IČO odběratelů
i dodavatelů, kódy odběrných míst (EAN, EIC) a fakturované částky. Vše je
uloženo na Google Disku, přístup byl poskytnut vedoucímu a oponentovi práce.

Bez těchto souborů nelze spustit pipeline nad reálnými fakturami. **Všechny
výsledky uvedené v práci ale reprodukovat lze** — reportovací notebooky
v [`experiments/notebooks/`](../experiments/notebooks/) pracují s předpočítanými
a anonymizovanými tabulkami v [`experiments/data/`](../experiments/data/), které
neobsahují žádnou hodnotu z faktury ani název souboru.

### Jak data doplnit

Stáhněte z Google Disku a rozbalte tak, aby vznikla tato struktura:

```text
data/ds1/<komodita>/<dodavatel>/<třída kvality>/<faktura>.pdf
data/ds1/ground_truth_<komodita>.csv
data/ds2/…                       stejné uspořádání
data/sorted/…                    referenční korpus pro kapitolu 3
```

Složky jsou uvedené v `.gitignore`, takže je nelze omylem commitnout.

## Sady

| Sada | Účel | Rozsah | Zdroj | Kvalita |
|---|---|---|---|---|
| **DS1** | finální evaluace (held-out) | 573 faktur, 19 dodavatelů | reálné faktury KEM | 50/36/14 % (přirozená) |
| **DS2** | vývoj a ladění promptů | 180 faktur, 8 dodavatelů | reálné faktury KEM | 33/33/33 % (uniformní) |
| **DS3** | test generalizace (zero-shot) | 162 faktur, 18 fiktivních dodavatelů | generováno `reportlab` | 33/33/33 % (uniformní) |

DS1 nebyla použita v žádné fázi vývoje ani ladění — slouží výhradně k závěrečné
evaluaci, aby byly výsledky objektivní. Rozdělení podle komodit je 320 / 45 / 55 /
23 / 75 / 55 (elektřina NN a VN, plyn MO a VO, teplo, voda).

### Sada DS3

Syntetická sada testuje, jak dobře model zobecňuje na dodavatele, které nikdy
neviděl. Žádný z 18 fiktivních dodavatelů se nevyskytuje v trénovacích datech
testovaných modelů, takže jde o čistý zero-shot test.

Každý dodavatel má jeden ze tří vizuálních stylů (minimalistický, strukturovaný,
dopisový) a devět faktur rovnoměrně rozdělených do tří stupňů umělé degradace.
Ground truth vzniká programově souběžně s fakturou, a je proto 100 % přesné.

Sadu lze vygenerovat znovu:

```bash
python scripts/generate_synthetic_invoices.py
```

## Třídy kvality

Q1 / Q2 / Q3 vycházejí z terciálů Laplaceovy variance (ostrosti obrazu) — postup
je popsán v notebooku
[`01_analyza_dat`](../experiments/notebooks/01_analyza_dat.ipynb).

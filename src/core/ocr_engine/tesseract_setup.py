"""
Nalezení spustitelného souboru Tesseractu nezávisle na operačním systému.

Na Linuxu a macOS i v kontejneru je Tesseract obvykle na ``PATH`` a stačí jej
nechat být. Na Windows se ale instaluje do ``Program Files`` a na ``PATH`` se
běžně nedostane, takže je potřeba cestu doplnit.

Pořadí hledání:

1. cesta předaná argumentem,
2. proměnná prostředí ``TESSERACT_CMD``,
3. ``PATH`` (běžný stav na Linuxu, macOS i v kontejneru),
4. obvyklá umístění instalátoru na Windows.

Nenajde-li se nic, funkce nastavení nemění a vrátí ``None``. Volání pak selže
až při skutečném použití OCR, a to s vlastní hláškou pytesseractu.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

__all__ = ["nastav_tesseract"]

# Kam instalátor pro Windows nejčastěji ukládá spustitelný soubor
_WINDOWS_UMISTENI = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def nastav_tesseract(cesta: str | os.PathLike[str] | None = None) -> str | None:
    """Nastaví ``pytesseract.tesseract_cmd`` a vrátí použitou cestu.

    Vrací ``None``, pokud Tesseract nebyl nalezen — volající se tak může
    rozhodnout, zda pokračovat, nebo skončit s vlastní hláškou.
    """
    import pytesseract

    kandidati: list[str] = []
    if cesta:
        kandidati.append(str(cesta))
    z_prostredi = os.getenv("TESSERACT_CMD")
    if z_prostredi:
        kandidati.append(z_prostredi)

    na_path = shutil.which("tesseract")
    if na_path:
        kandidati.append(na_path)

    if sys.platform == "win32":
        kandidati.extend(_WINDOWS_UMISTENI)

    for kandidat in kandidati:
        if Path(kandidat).is_file() or shutil.which(kandidat):
            pytesseract.pytesseract.tesseract_cmd = kandidat
            return kandidat

    return None

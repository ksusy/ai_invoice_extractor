# -*- coding: utf-8 -*-
"""
Jednotný vizuální styl grafů pro reportovací notebooky.

Modul drží stejnou paletu, typografii i kompozici, jaká je použita v obrázcích
bakalářské práce, takže grafy z notebooků a grafy v textu práce tvoří jeden
vizuální celek.

Barevná konvence
----------------
* **Komodity** — kvalitativní paleta ColorBrewer Set2, pevně přiřazená ke
  komoditě. Barva se nikdy nepřebarvuje podle pořadí ve výsledku, takže
  „Teplo“ je zelené ve všech grafech napříč všemi notebooky.
* **Srovnání metod** — tmavě modrá zvýrazňuje zvolenou (vítěznou) variantu,
  červená protikandidáta, ostatní zůstávají neutrálně šedé. Graf tak sděluje
  rozhodnutí, ne jen čísla.

Paleta Set2 je pastelová a samotný odstín by nestačil jako jediný nositel
informace. Grafy proto identitu vždy kódují ještě druhým kanálem — popiskem
osy nebo přímou hodnotou u sloupce — a barva slouží jako doplňkové vodítko.

Použití
-------
    import sys; sys.path.insert(0, "..")
    from thesis_style import *
    pouzij_styl()
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

__all__ = [
    "BARVY_KOMODIT", "NAZVY_KOMODIT", "KOMODITY",
    "VITEZ", "PORAZENY", "NEUTRALNI", "PRAH",
    "pouzij_styl", "barvy_vitez", "popisky_sloupcu", "uprav_osu",
    "cara_prahu", "popisky_panelu", "tabulka", "uloz",
]

# ── Komodity ────────────────────────────────────────────────────────────────
BARVY_KOMODIT: dict[str, str] = {
    "elektrina_nn": "#8DA0CB",   # modrá
    "elektrina_vn": "#E78AC3",   # růžová
    "plyn_mo":      "#FC8D62",   # oranžová
    "plyn_vo":      "#FFD92F",   # žlutá
    "teplo":        "#A6D854",   # zelená
    "voda":         "#66C2A5",   # tyrkysová
}

NAZVY_KOMODIT: dict[str, str] = {
    "elektrina_nn": "Elektřina NN",
    "elektrina_vn": "Elektřina VN",
    "plyn_mo":      "Plyn MO",
    "plyn_vo":      "Plyn VO",
    "teplo":        "Teplo",
    "voda":         "Voda",
}

KOMODITY: list[str] = list(BARVY_KOMODIT)

# ── Srovnání variant ────────────────────────────────────────────────────────
VITEZ     = "#1B4F72"   # zvolená / produkční varianta
PORAZENY  = "#E74C3C"   # varianta, která neuspěla
NEUTRALNI = "#BDBDBD"   # ostatní kandidáti
PRAH      = "#C0392B"   # čára akceptačního prahu


def pouzij_styl() -> None:
    """Nastaví globální téma matplotlibu shodné s obrázky v práci."""
    sns.set_theme(style="whitegrid", context="notebook")
    sns.set_palette(list(BARVY_KOMODIT.values()))

    plt.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "white",
        "axes.edgecolor":     "#CCCCCC",
        "axes.linewidth":     0.8,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.titleweight":   "semibold",
        "axes.titlesize":     13,
        "axes.labelsize":     11,
        "axes.axisbelow":     True,
        "axes.grid":          True,
        "grid.color":         "#F3F3F3",
        "grid.linestyle":     "--",
        "grid.linewidth":     0.6,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.major.size":   4,
        "ytick.major.size":   4,
        "legend.fontsize":    10,
        "legend.frameon":     False,
        "figure.titlesize":   13,
        "figure.dpi":         110,
        "savefig.dpi":        200,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.1,
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Libertinus Sans", "DejaVu Sans", "Arial", "Helvetica"],
    })


def barvy_vitez(popisky: Sequence[str], vitez: str,
                porazeny: str | None = None) -> list[str]:
    """Barvy sloupců pro srovnání variant: vítěz tmavě, poražený červeně."""
    out = []
    for popisek in popisky:
        if popisek == vitez:
            out.append(VITEZ)
        elif porazeny is not None and popisek == porazeny:
            out.append(PORAZENY)
        else:
            out.append(NEUTRALNI)
    return out


def popisky_sloupcu(ax, fmt: str = "{:.3f}", odsazeni: float = 3.0,
                    velikost: int = 9, vodorovne: bool = False) -> None:
    """Vypíše hodnotu ke každému sloupci — druhý nositel informace vedle barvy."""
    for patch in ax.patches:
        if vodorovne:
            hodnota = patch.get_width()
            if not np.isfinite(hodnota):
                continue
            ax.annotate(fmt.format(hodnota),
                        (hodnota, patch.get_y() + patch.get_height() / 2),
                        ha="left", va="center", xytext=(odsazeni, 0),
                        textcoords="offset points", fontsize=velikost,
                        color="#333333")
        else:
            hodnota = patch.get_height()
            if not np.isfinite(hodnota):
                continue
            ax.annotate(fmt.format(hodnota),
                        (patch.get_x() + patch.get_width() / 2, hodnota),
                        ha="center", va="bottom", xytext=(0, odsazeni),
                        textcoords="offset points", fontsize=velikost,
                        color="#333333")


def uprav_osu(ax, nadpis: str | None = None, x: str | None = None,
              y: str | None = None, legenda: bool = False,
              umisteni: str = "best"):
    """Sjednocené dokončení osy — nadpis, popisky, legenda."""
    if nadpis:
        ax.set_title(nadpis, fontsize=13, fontweight="semibold")
    if x is not None:
        ax.set_xlabel(x)
    if y is not None:
        ax.set_ylabel(y)
    if legenda and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=umisteni, frameon=False)
    return ax


def cara_prahu(ax, hodnota: float, popis: str, vodorovne: bool = True) -> None:
    """Čárkovaná čára akceptačního prahu s popiskem v legendě."""
    kresli = ax.axhline if vodorovne else ax.axvline
    kresli(hodnota, color=PRAH, linestyle="--", linewidth=1.2,
           label=popis, zorder=1)
    ax.legend(frameon=False, fontsize=9)


def popisky_panelu(axes, popisky: Sequence[str] | None = None) -> None:
    """Označí panely (a), (b), (c) … pro odkazování z textu."""
    ploche = np.array(axes).ravel()
    if popisky is None:
        popisky = [f"({chr(97 + i)})" for i in range(len(ploche))]
    for ax, popisek in zip(ploche, popisky):
        ax.text(-0.08, 1.06, popisek, transform=ax.transAxes,
                fontsize=11, fontweight="semibold", va="top")


def tabulka(df: pd.DataFrame, zvyrazni: Mapping[str, str] | None = None,
            desetinna: int = 3):
    """Čitelná tabulka pod grafem — textová alternativa k barevnému kódování."""
    styl = (df.style
              .format(precision=desetinna)
              .set_table_styles([
                  {"selector": "th",
                   "props": [("background-color", "#F5F5F5"),
                             ("font-weight", "600"),
                             ("text-align", "left"),
                             ("border-bottom", "1px solid #CCCCCC")]},
                  {"selector": "td", "props": [("text-align", "right")]},
              ]))
    if zvyrazni:
        for sloupec, barva in zvyrazni.items():
            if sloupec in df.columns:
                styl = styl.background_gradient(subset=[sloupec], cmap=barva)
    return styl


def uloz(fig, nazev: str, adresar: str | Path = "obrazky") -> Path:
    """Uloží obrázek do složky notebooku a vrátí cestu."""
    cesta = Path(adresar)
    cesta.mkdir(parents=True, exist_ok=True)
    soubor = cesta / (nazev if nazev.endswith(".png") else f"{nazev}.png")
    fig.savefig(soubor)
    return soubor

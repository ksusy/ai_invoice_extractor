"""
Central styling module for experiments.
Provides a consistent visual theme, commodity colors, and output utilities.

ColorBrewer Set2 palette + consistent colors for energy commodities.
Academic, clean design without decorative noise.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ============================================================================
# COMMODITY COLORS - ColorBrewer Set2 + energetické komodity
# ============================================================================

COMM_COLORS = {
    'elektrina_nn': '#8DA0CB',  # Modrá
    'elektrina_vn': '#E78AC3',  # Růžová
    'plyn_mo':      '#FC8D62',  # Oranžová
    'plyn_vo':      '#FFD92F',  # Žlutá
    'teplo':        '#A6D854',  # Zelená
    'voda':         '#66C2A5',  # Tyrkysová
}

COMM_LABELS = {
    'elektrina_nn': 'Elektřina NN',
    'elektrina_vn': 'Elektřina VN',
    'plyn_mo':      'Plyn MO',
    'plyn_vo':      'Plyn VO',
    'teplo':        'Teplo',
    'voda':         'Voda',
}

COMM_KEYS = list(COMM_COLORS.keys())


# ============================================================================
# WINNER / LOSER / NEUTRAL - jednotné zvýraznění pro srovnávací grafy
# (metoda × metoda, ne komodita × komodita) - stejná "rodina" jako Set2 výše.
# ============================================================================

WINNER_COLOR  = '#1B4F72'   # tmavě modrá - vítěz / produkční volba
LOSER_COLOR   = '#E74C3C'   # červená - alternativa, která prohrála
NEUTRAL_COLOR = '#BDBDBD'   # světle šedá - ostatní kandidáti v žebříčku


def winner_bar_colors(labels: Sequence[str], winner: str, loser: str | None = None) -> list[str]:
    """Vrátí barvy pro sloupcový graf, kde je vítěz zvýrazněn tmavě modře.

    Pokud je zadán `loser`, zvýrazní se červeně; ostatní jsou neutrálně šedé.
    """
    out = []
    for label in labels:
        if label == winner:
            out.append(WINNER_COLOR)
        elif loser is not None and label == loser:
            out.append(LOSER_COLOR)
        else:
            out.append(NEUTRAL_COLOR)
    return out


# ============================================================================
# MATPLOTLIB AKADEMICKÝ THEME
# ============================================================================

def get_commodity_palette(grayscale: bool = False, commodity_keys: Sequence[str] | None = None) -> dict[str, str]:
    """Return a palette mapping for commodity keys."""
    keys = list(commodity_keys) if commodity_keys is not None else COMM_KEYS
    if not grayscale:
        return {key: COMM_COLORS.get(key, '#808080') for key in keys}

    shades = sns.color_palette('Greys', n_colors=max(len(keys) + 2, 3))
    usable = shades[1 : 1 + len(keys)]
    return {key: mcolors.to_hex(color) for key, color in zip(keys, usable, strict=False)}


def apply_academic_theme(grayscale: bool = False):
    """
    Set a global matplotlib theme for academic figures.
    Clean design with subtle grid and publication-friendly defaults.
    """
    palette = get_commodity_palette(grayscale=grayscale)

    sns.set_theme(style='whitegrid', context='notebook')
    sns.set_palette(list(palette.values()))

    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = '#CCCCCC'
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.titleweight'] = 'semibold'
    plt.rcParams['legend.frameon'] = False

    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Libertinus Sans', 'DejaVu Sans', 'Arial', 'Helvetica']

    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 13

    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = '#F3F3F3'
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.6
    plt.rcParams['axes.axisbelow'] = True

    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['xtick.minor.size'] = 2
    plt.rcParams['ytick.minor.size'] = 2

    plt.rcParams['figure.dpi'] = 110
    plt.rcParams['savefig.dpi'] = 150
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1


def finalize_axis(
    ax,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = False,
    legend_loc: str = 'best',
):
    """Apply a consistent finishing pass to an axis."""
    if title:
        ax.set_title(title, fontsize=13, fontweight='semibold')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc=legend_loc, frameon=False)
    return ax


def annotate_bars(ax, fmt: str = '{:.2f}', padding: float = 3.0, rotation: int = 0):
    """Annotate bar heights with numeric labels."""
    for bar in ax.patches:
        height = bar.get_height()
        if np.isnan(height):
            continue
        x = bar.get_x() + bar.get_width() / 2
        ax.annotate(
            fmt.format(height),
            (x, height),
            ha='center',
            va='bottom',
            xytext=(0, padding),
            textcoords='offset points',
            rotation=rotation,
            fontsize=9,
        )


def add_panel_labels(axes, labels: Sequence[str] | None = None, x_offset: float = -0.08, y_offset: float = 1.04):
    """Add panel labels such as (a), (b), (c) to a subplot grid."""
    flat_axes = np.array(axes).ravel()
    if labels is None:
        labels = [f'({chr(97 + idx)})' for idx in range(len(flat_axes))]
    for ax, label in zip(flat_axes, labels, strict=False):
        ax.text(
            x_offset,
            y_offset,
            label,
            transform=ax.transAxes,
            fontsize=11,
            fontweight='semibold',
            va='top',
        )


def plot_bar_with_error(
    data: pd.DataFrame,
    x: str,
    y: str,
    yerr: str | None = None,
    ax=None,
    color: str | None = None,
    palette: Mapping[str, str] | None = None,
    order: Sequence[Any] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    annotate: bool = True,
):
    """Render a publication-style bar chart with optional error bars."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    plot_data = data.copy()
    if order is not None:
        plot_data = plot_data.set_index(x).reindex(order).reset_index()

    bar_colors = None
    if palette is not None:
        bar_colors = [palette.get(label, color or '#4C72B0') for label in plot_data[x]]
    elif color is not None:
        bar_colors = color

    positions = np.arange(len(plot_data))
    ax.bar(positions, plot_data[y].to_numpy(), color=bar_colors)
    if yerr is not None:
        ax.errorbar(
            positions,
            plot_data[y].to_numpy(),
            yerr=plot_data[yerr].to_numpy(),
            fmt='none',
            ecolor='#333333',
            elinewidth=1.0,
            capsize=4,
        )
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_data[x].astype(str).tolist(), rotation=0)
    if annotate:
        annotate_bars(ax)
    return finalize_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel)


def plot_box_violin(
    data: pd.DataFrame,
    x: str,
    y: str,
    ax=None,
    kind: str = 'box',
    hue: str | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    inner: str = 'quartile',
    palette: Mapping[str, str] | None = None,
):
    """Render a box or violin plot with the shared academic style."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    if kind == 'violin':
        sns.violinplot(data=data, x=x, y=y, hue=hue, ax=ax, inner=inner, palette=palette, cut=0)
    else:
        sns.boxplot(data=data, x=x, y=y, hue=hue, ax=ax, palette=palette, showfliers=False)

    return finalize_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel, legend=hue is not None)


def plot_heatmap(
    data: pd.DataFrame,
    ax=None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str = 'Blues',
    annot: bool = True,
    fmt: str = '.2f',
    cbar_kws: Mapping[str, Any] | None = None,
    linewidths: float = 0.5,
):
    """Render a heatmap suitable for correlation or score tables."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    sns.heatmap(data, ax=ax, cmap=cmap, annot=annot, fmt=fmt, cbar_kws=cbar_kws, linewidths=linewidths)
    return finalize_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel)


def plot_scatter_with_regression(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    ax=None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    scatter_kws: Mapping[str, Any] | None = None,
    line_kws: Mapping[str, Any] | None = None,
):
    """Render a scatter plot with an optional regression line."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    if hue is not None:
        sns.scatterplot(data=data, x=x, y=y, hue=hue, ax=ax, **dict(scatter_kws or {}))
        ax.legend(frameon=False)
        sns.regplot(
            data=data,
            x=x,
            y=y,
            ax=ax,
            scatter=False,
            line_kws={'color': '#222222', 'linewidth': 1.8, **dict(line_kws or {})},
            ci=95,
            truncate=False,
        )
    else:
        sns.regplot(
            data=data,
            x=x,
            y=y,
            ax=ax,
            scatter_kws={'s': 25, 'alpha': 0.7, **dict(scatter_kws or {})},
            line_kws={'color': '#222222', 'linewidth': 1.8, **dict(line_kws or {})},
            ci=95,
            truncate=False,
        )

    return finalize_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel, legend=hue is not None)


def plot_cdf(
    data: pd.DataFrame,
    value_col: str,
    group_col: str | None = None,
    ax=None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = 'Cumulative share',
    palette: Mapping[str, str] | None = None,
):
    """Render a cumulative distribution function plot."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))

    if group_col is None:
        values = np.sort(data[value_col].dropna().to_numpy())
        if len(values) > 0:
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), linewidth=2)
    else:
        palette = dict(palette or {})
        for group_name, group_df in data.groupby(group_col, dropna=False):
            values = np.sort(group_df[value_col].dropna().to_numpy())
            if len(values) == 0:
                continue
            color = palette.get(group_name, None)
            ax.plot(values, np.arange(1, len(values) + 1) / len(values), label=str(group_name), linewidth=2, color=color)
        ax.legend(frameon=False)

    return finalize_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel, legend=group_col is not None)


def plot_missingness_heatmap(
    data: pd.DataFrame,
    ax=None,
    title: str | None = None,
    cmap: str = 'mako',
):
    """Render a missing-value heatmap for quick data quality checks."""
    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, min(18, data.shape[1] * 0.7)), 5))

    missing = data.isna().astype(int)
    sns.heatmap(missing, ax=ax, cmap=cmap, cbar=False, yticklabels=False)
    return finalize_axis(ax, title=title, xlabel='Columns', ylabel='Rows')


def style_dataframe(df):
    """
    Styl pro pandas DataFrame v notebooku.
    Čistý design s bílým pozadím a modrými záhlavím.
    """
    return df.style.set_properties(**{
        'background-color': '#FAFAFA',
        'border-color': '#E0E0E0',
        'font-size': '10.5pt',
        'font-family': 'Libertinus Sans, Arial',
        'text-align': 'right',
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#8DA0CB'),  # Modrá z Set2
            ('color', 'white'),
            ('font-weight', 'bold'),
            ('text-align', 'center'),
            ('font-family', 'Libertinus Sans, Arial'),
            ('font-size', '11pt'),
        ]},
        {'selector': 'td', 'props': [
            ('padding', '8px'),
            ('border', '1px solid #E0E0E0'),
        ]},
    ]).format(na_rep='N/A')


# ============================================================================
# UTILITY FUNKCE PRO UKLÁDÁNÍ VÝSTUPŮ
# ============================================================================

def get_figure_dir(base_results_dir='./results/figures'):
    """Vrátí cestu k adresáři s figurami, vytvoří ji pokud neexistuje."""
    path = Path(base_results_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_dir(base_results_dir='./results/data'):
    """Vrátí cestu k adresáři s daty, vytvoří ji pokud neexistuje."""
    path = Path(base_results_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_reports_dir(base_results_dir='./results/reports'):
    """Vrátí cestu k adresáři s reporty, vytvoří ji pokud neexistuje."""
    path = Path(base_results_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig, filename, results_dir='./results/figures', metadata=None):
    """
    Uloží figurku s metadaty (JSON vedle).
    
    Args:
        fig: matplotlib figure
        filename: jméno souboru bez přípony (png se přidá automaticky)
        results_dir: cesta k results adresáři
        metadata: dict s metadaty (bude uložen jako JSON)
    
    Returns:
        Path k uloženému souboru
    """
    fig_dir = get_figure_dir(results_dir)
    fig_path = fig_dir / f"{filename}.png"

    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight', facecolor='white')

    if metadata:
        metadata = dict(metadata)
        metadata.setdefault('description', filename.replace('_', ' '))
        metadata_path = fig_dir / f"{filename}_metadata.json"
        metadata['saved_at'] = datetime.now().isoformat()
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return fig_path


def save_dataframe(df, filename, results_dir='./results/data', format='csv', metadata=None):
    """
    Uloží DataFrame s metadaty.
    
    Args:
        df: pandas DataFrame
        filename: jméno souboru bez přípony
        results_dir: cesta k results adresáři
        format: 'csv' nebo 'parquet'
        metadata: dict s metadaty
    
    Returns:
        Path k uloženému souboru
    """
    data_dir = get_data_dir(results_dir)
    
    if format == 'csv':
        data_path = data_dir / f"{filename}.csv"
        df.to_csv(data_path, index=False, encoding='utf-8')
    elif format == 'parquet':
        data_path = data_dir / f"{filename}.parquet"
        df.to_parquet(data_path, index=False)
    else:
        raise ValueError(f"Neznámý formát: {format}")
    
    if metadata:
        metadata = dict(metadata)
        metadata.setdefault('description', filename.replace('_', ' '))
        metadata_path = data_dir / f"{filename}_metadata.json"
        metadata['shape'] = df.shape
        metadata['columns'] = df.columns.tolist()
        metadata['saved_at'] = datetime.now().isoformat()
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    return data_path


def log_experiment(experiment_name, results_dir='./results/logs'):
    """
    Vrátí logger pro experiment. Loguje do souboru i konzole.
    
    Args:
        experiment_name: jméno experimentu
        results_dir: cesta k results adresáři
    
    Returns:
        logging.Logger instance
    """
    log_dir = Path(results_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()
    
    fh = logging.FileHandler(log_dir / f"{experiment_name}.log", encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# UTILITY FUNKCE PRO GRAFY
# ============================================================================

def get_commodity_color(commodity_key):
    """Vrátí barvu komodity ze sady Set2."""
    return COMM_COLORS.get(commodity_key, '#808080')


def get_commodity_label(commodity_key):
    """Vrátí popisek komodity."""
    return COMM_LABELS.get(commodity_key, commodity_key)


def format_axis_pct(ax, axis='y'):
    """Formátuje osu jako procenta bez zbytečných nul."""
    if axis == 'y':
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    else:
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))


def format_axis_thousands(ax, axis='y'):
    """Formátuje osu se separátorem tisíců."""
    if axis == 'y':
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{int(x):,}'))
    else:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{int(x):,}'))


# Inicializace
apply_academic_theme()


__all__ = [
    'COMM_COLORS',
    'COMM_LABELS',
    'COMM_KEYS',
    'WINNER_COLOR',
    'LOSER_COLOR',
    'NEUTRAL_COLOR',
    'winner_bar_colors',
    'apply_academic_theme',
    'get_commodity_palette',
    'style_dataframe',
    'get_figure_dir',
    'get_data_dir',
    'get_reports_dir',
    'save_figure',
    'save_dataframe',
    'log_experiment',
    'get_commodity_color',
    'get_commodity_label',
    'format_axis_pct',
    'format_axis_thousands',
    'finalize_axis',
    'annotate_bars',
    'add_panel_labels',
    'plot_bar_with_error',
    'plot_box_violin',
    'plot_heatmap',
    'plot_scatter_with_regression',
    'plot_cdf',
    'plot_missingness_heatmap',
]

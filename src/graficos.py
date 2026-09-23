"""Estilo visual único para todos os gráficos do projeto."""

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from src.dados import RAIZ

PASTA_FIGURAS = RAIZ / "reports" / "figures"

# Paleta categórica validada para daltonismo (slots 1 e 2) + tons neutros
AZUL = "#2a78d6"        # clientes ativos / série principal
LARANJA = "#eb6834"     # cancelados / destaque de risco
AZUL_CLARO = "#86b6ef"
SUPERFICIE = "#fcfcfb"
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"
GRADE = "#e4e3df"
CORES_STATUS = {"Cliente": AZUL, "Cancelado": LARANJA}


def aplicar_estilo() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": TEXTO,
        "axes.labelcolor": TEXTO_SECUNDARIO,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 22,
        "axes.edgecolor": GRADE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.8,
        "xtick.color": TEXTO_SECUNDARIO,
        "ytick.color": TEXTO_SECUNDARIO,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "lines.linewidth": 2,
    })


def eixo_percentual(ax, eixo: str = "y") -> None:
    alvo = ax.yaxis if eixo == "y" else ax.xaxis
    alvo.set_major_formatter(PercentFormatter(1.0, decimals=0))


def linha_media(ax, valor: float, horizontal: bool = True) -> None:
    """Linha tracejada de referência; o valor é citado no subtítulo do gráfico."""
    kwargs = dict(color=TEXTO_SECUNDARIO, linestyle="--", linewidth=1)
    (ax.axhline if horizontal else ax.axvline)(valor, **kwargs)


def subtitulo(ax, texto: str) -> None:
    ax.text(0, 1.01, texto, transform=ax.transAxes, fontsize=9.5, color=TEXTO_SECUNDARIO, va="bottom")


def salvar(fig, nome: str) -> None:
    PASTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PASTA_FIGURAS / f"{nome}.png")

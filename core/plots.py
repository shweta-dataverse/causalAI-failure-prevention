from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# categorical colours, always used in this fixed order (colour-blind checked)
COLORS: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")

# ink and chrome for text, axes and grid
INK: dict[str, str] = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "surface": "#ffffff",
}

# one sequential hue (blue), light -> dark
SEQUENTIAL_BLUE: tuple[str, ...] = (
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
)

SUBSYSTEM_COLORS = {"coolant": COLORS[0], "hydraulics": COLORS[1], "probe": COLORS[2]}
SOURCE_COLORS = {"real_op": COLORS[0], "dig_twin": COLORS[1]}
SOURCE_LABELS = {"real_op": "normal operation", "dig_twin": "fault runs"}
TYPE_COLORS = {
    "Binary": COLORS[0],
    "Alarm": COLORS[1],
    "Continuous": COLORS[2],
    "Counter": COLORS[3],
    "Categorical": COLORS[4],
}
GRAPH_COLORS = {"expert": COLORS[0], "discovered": COLORS[1]}
GRAPH_LABELS = {"expert": "Expert graph", "discovered": "Discovered graph"}
BASELINE_COLOR = "#8A8A8A"

# text width of the printed page in inches, so 10 pt in a figure is 10 pt in print
PAGE_WIDTH = 5.5


def set_plot_style() -> None:
    """quiet style: white background, light grid, no top/right spines."""
    plt.rcParams.update(
        {
            "figure.facecolor": INK["surface"],
            "axes.facecolor": INK["surface"],
            "savefig.facecolor": INK["surface"],
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.labelcolor": INK["secondary"],
            "axes.edgecolor": INK["axis"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.prop_cycle": plt.cycler(color=list(COLORS)),
            "grid.color": INK["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": INK["secondary"],
            "ytick.color": INK["secondary"],
            "text.color": INK["primary"],
            "lines.linewidth": 2,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def set_print_style() -> None:
    """the base style at print size, as used for every figure of the document."""
    set_plot_style()
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
                         "axes.labelcolor": INK["primary"], "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
                         "legend.fontsize": 9.5, "xtick.color": INK["primary"], "ytick.color": INK["primary"]})


def sequential_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("sequential_blue", list(SEQUENTIAL_BLUE))


def write_pdf(fig: plt.Figure, file: Path, dpi: int = 200) -> Path:
    """save a figure as pdf and close it."""
    file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(file, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return file

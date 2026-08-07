"""Plot the mini cheetah single-leg reference trajectory."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

CSV = (
    Path(__file__).resolve().parents[1]
    / "resources/robots/mini_cheetah/trajectories/single_leg.csv"
)

COLORS = {"HAA": "#2a78d6", "HFE": "#eb6834", "KFE": "#1baf7a"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    names = [n.strip() for n in rows[0]]
    cols = {n: [] for n in names}
    for row in rows[1:]:
        if not any(c.strip() for c in row):
            continue
        for n, v in zip(names, row):
            cols[n].append(float(v))
    return names, cols


def main():
    names, cols = load(CSV)
    steps = range(len(cols[names[0]]))

    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for name in names:
        y = cols[name]
        ax.plot(steps, y, lw=2, color=COLORS[name], label=name, zorder=3)
        ax.annotate(
            name,
            xy=(len(y) - 1, y[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            color=INK,
        )

    ax.set_title(
        "Mini cheetah single-leg reference trajectory", color=INK, fontsize=13, pad=12
    )
    ax.set_xlabel("Frame", color=MUTED)
    ax.set_ylabel("Joint position (rad)", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.tick_params(colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.legend(frameon=False, labelcolor=INK, loc="upper right", ncols=3)
    ax.margins(x=0.06)

    fig.tight_layout()
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")
    plt.show()


if __name__ == "__main__":
    main()

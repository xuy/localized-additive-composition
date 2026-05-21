#!/usr/bin/env python3
"""Regenerate paper Figure 1 from cached experiment outputs in ../../results/.

Usage:
    python paper/figures/regen_figure.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent.parent
RESULTS = REPO / "results"
OUT = Path(__file__).resolve().parent


def plot_panel(ax, path, title, layers):
    d = json.load(open(path))
    pos_colors = {"p_last": "tab:blue", "g1": "tab:orange", "g2": "tab:green"}
    pos_labels = {"p_last": r"$p_{\text{last}}$", "g1": "$g_1$", "g2": "$g_2$"}
    markers = {"p_last": "o", "g1": "s", "g2": "^"}
    for pos in ["p_last", "g1", "g2"]:
        ys, ys_lo, ys_hi = [], [], []
        for L in layers:
            s = d["causal"][pos][str(L)]["summary"]
            ys.append(s["median"])
            ys_lo.append(s["iqr"][0])
            ys_hi.append(s["iqr"][1])
        ax.plot(layers, ys, color=pos_colors[pos], marker=markers[pos],
                label=pos_labels[pos], linewidth=2, markersize=7)
        ax.fill_between(layers, ys_lo, ys_hi, color=pos_colors[pos], alpha=0.13)
    ax.set_xlabel("layer L")
    ax.set_ylabel("median causal KL (10-token window)")
    ax.set_title(title)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylim(0, None)
    ax.axhline(0.1, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.5))

    plot_panel(axes[0, 0], RESULTS / "v15_gemma2b_localized_positions.json",
               "Gemma-2-2B-IT (short grid, 12 cells)", [6, 10, 14, 18, 22])
    plot_panel(axes[0, 1], RESULTS / "v15_qwen15_localized_positions.json",
               "Qwen-2.5-1.5B (short grid, 12 cells)", [7, 11, 15, 20, 24])
    plot_panel(axes[1, 0], RESULTS / "v16_gemma2b_diverse_grid.json",
               "Gemma-2-2B-IT (long-persona grid, 48 cells)", [10, 14, 18])
    plot_panel(axes[1, 1], RESULTS / "v16_qwen15_diverse_grid.json",
               "Qwen-2.5-1.5B (long-persona grid, 48 cells)", [7, 15, 20])

    plt.tight_layout()
    plt.savefig(OUT / "localized_kl_by_layer.pdf", bbox_inches="tight")
    plt.savefig(OUT / "localized_kl_by_layer.png", dpi=160, bbox_inches="tight")
    print(f"Saved {OUT / 'localized_kl_by_layer.pdf'}")
    print(f"Saved {OUT / 'localized_kl_by_layer.png'}")


if __name__ == "__main__":
    main()

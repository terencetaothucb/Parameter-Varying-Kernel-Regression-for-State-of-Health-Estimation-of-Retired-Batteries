from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from observer_benchmark_utils import conditional_linearity_rows, load_aggregated_20ah


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = REPO_ROOT / "figures" / "fixed_soc_conditional_observability.png"
OUT_PDF = REPO_ROOT / "figures" / "fixed_soc_conditional_observability.pdf"


def main() -> None:
    data = load_aggregated_20ah()
    cond = pd.DataFrame(conditional_linearity_rows(data))

    selected_soc = [20, 35, 50, 65]
    selected = cond[cond["soc_knot"].isin(selected_soc)].copy()

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8), constrained_layout=True)
    axes = axes.ravel()
    colors = ["#1d3557", "#457b9d", "#2a9d8f", "#e76f51"]

    for ax, color, (_, row) in zip(axes, colors, selected.iterrows()):
        soc_knot = float(row["soc_knot"])
        feature = row["best_feature"]
        corr = float(row["best_corr"])
        feat_idx = data.feature_labels.index(feature)
        mask = (data.source == "fixed") & np.isclose(data.soc, soc_knot)
        x = data.soh[mask]
        y = 1000.0 * data.features[mask, feat_idx]

        pulse_width, feature_name = feature.split("_Hyst_M3_")
        title = f"SOC {int(soc_knot)}%-{feature_name}-{pulse_width} ms"

        ax.scatter(x, y, s=28, alpha=0.82, color=color, edgecolors="none")
        coef = np.polyfit(x, y, 1)
        xline = np.linspace(x.min(), x.max(), 100)
        yline = coef[0] * xline + coef[1]
        ax.plot(xline, yline, color="#222222", linewidth=2.0)

        ax.set_title(title)
        ax.set_xlabel(r"$x=\mathrm{SOH}$")
        ax.text(
            0.03,
            0.97,
            f"$\\rho={corr:.3f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
        )
        ax.grid(True, alpha=0.25, linewidth=0.8)

    axes[0].set_ylabel(r"$y_i$ (mV)")
    axes[2].set_ylabel(r"$y_i$ (mV)")

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()

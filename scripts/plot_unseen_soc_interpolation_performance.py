from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "results" / "benchmark_results_20ah.xlsx"
OUT_PNG = REPO_ROOT / "figures" / "unseen_soc_interpolation_performance.png"
OUT_PDF = REPO_ROOT / "figures" / "unseen_soc_interpolation_performance.pdf"


def main() -> None:
    df = pd.read_excel(RESULTS_PATH, sheet_name="HardSOCInterpolation")
    ordered_models = [
        "B0-Global-H-Linear",
        "B1-Global-HS-Linear",
        "B2-Global-H-Poly",
        "B3-Global-HS-Poly",
        "B4-Discrete-Local",
        "Ours",
    ]
    df = df[df["model"].isin(ordered_models)].copy()

    protocol_to_case = {
        "hold_6_alt": "Case 1",
        "hold_7_dense": "Case 2",
        "hold_8_dense": "Case 3",
    }
    model_labels = {
        "B0-Global-H-Linear": "B0",
        "B1-Global-HS-Linear": "B1",
        "B2-Global-H-Poly": "B2",
        "B3-Global-HS-Poly": "B3",
        "B4-Discrete-Local": "B4",
        "Ours": "Ours",
    }
    ordered_protocols = ["hold_6_alt", "hold_7_dense", "hold_8_dense"]

    pivot_r2 = (
        df.pivot(index="model", columns="protocol", values="r2")
        .loc[ordered_models, ordered_protocols]
        .rename(index=model_labels, columns=protocol_to_case)
    )
    pivot_mape = (
        df.pivot(index="model", columns="protocol", values="median_ape_pct")
        .loc[ordered_models, ordered_protocols]
        .rename(index=model_labels, columns=protocol_to_case)
    )

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(7.1, 4.0), constrained_layout=True)
    ax2 = ax.twinx()
    x = np.arange(len(pivot_r2.index))
    width = 0.22
    colors = ["#457b9d", "#2a9d8f", "#e76f51"]

    for i, case in enumerate(pivot_r2.columns):
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            pivot_r2[case].values,
            width=width,
            label=case,
            color=colors[i],
            edgecolor="white",
            linewidth=0.8,
        )
        ax2.scatter(
            x + offset,
            pivot_mape[case].values,
            s=42,
            color=colors[i],
            marker="o",
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )

    ax.set_ylabel(r"$R^2$")
    ax2.set_ylabel("Median APE (%)")
    ax.set_xlabel("Method")
    ax.set_xticks(x)
    ax.set_xticklabels(list(pivot_r2.index))
    ax.set_ylim(0.72, 0.89)
    ax2.set_ylim(1.1, 1.82)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(loc="upper left", ncol=3, frameon=False)

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")


if __name__ == "__main__":
    main()

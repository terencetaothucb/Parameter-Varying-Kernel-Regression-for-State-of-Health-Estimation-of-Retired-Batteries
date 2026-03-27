from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from observer_benchmark_utils import load_aggregated_20ah
from run_parameter_varying_observer_benchmarks import run_protocols


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = REPO_ROOT / "figures" / "pulse_amplitude_impact.png"
OUT_PDF = REPO_ROOT / "figures" / "pulse_amplitude_impact.pdf"
OUT_CSV = REPO_ROOT / "results" / "pulse_amplitude_impact_detail.csv"


def subset_by_rates(labels: list[str], rates: set[str]) -> list[int]:
    return [i for i, label in enumerate(labels) if any(label.endswith(f"Hyst_M3_{rate}") for rate in rates)]


def main() -> None:
    data = load_aggregated_20ah()
    labels = data.feature_labels

    subset_specs = {
        "Low amplitude": subset_by_rates(labels, {"0.5C", "1C"}),
        "Mid amplitude": subset_by_rates(labels, {"1.5C", "2C"}),
        "High amplitude": subset_by_rates(labels, {"2.5C"}),
    }
    protocol_to_case = {
        "hold_6_alt": "Case 1",
        "hold_7_dense": "Case 2",
        "hold_8_dense": "Case 3",
    }
    ordered_models = [
        "B0-Global-H-Linear",
        "B1-Global-HS-Linear",
        "B2-Global-H-Poly",
        "B3-Global-HS-Poly",
        "B4-Discrete-Local",
        "Ours",
    ]
    model_to_label = {
        "B0-Global-H-Linear": "B0",
        "B1-Global-HS-Linear": "B1",
        "B2-Global-H-Poly": "B2",
        "B3-Global-HS-Poly": "B3",
        "B4-Discrete-Local": "B4",
        "Ours": "Ours",
    }

    rows: list[dict[str, str | float | int]] = []
    for subset_name, keep in subset_specs.items():
        subset_data = replace(data, features=data.features[:, keep], feature_labels=[labels[i] for i in keep])
        protocol_rows, _, _ = run_protocols(subset_data)
        frame = pd.DataFrame(protocol_rows)
        frame = frame[frame["model"].isin(ordered_models)].copy()
        frame["subset"] = subset_name
        frame["case"] = frame["protocol"].map(protocol_to_case)
        frame["model_label"] = frame["model"].map(model_to_label)
        rows.extend(frame.to_dict("records"))

    detail = pd.DataFrame(rows)
    detail.to_csv(OUT_CSV, index=False)

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

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), constrained_layout=True, sharex=True)
    colors = {"Case 1": "#457b9d", "Case 2": "#2a9d8f", "Case 3": "#e76f51"}
    width = 0.22
    x = np.arange(len(ordered_models))
    subset_note = {
        "Low amplitude": "Low amplitude\n0.5C, 1C",
        "Mid amplitude": "Mid amplitude\n1.5C, 2C",
        "High amplitude": "High amplitude\n2.5C",
    }

    for ax, subset_name in zip(axes, subset_specs):
        sub = detail[detail["subset"] == subset_name]
        for i, case in enumerate(["Case 1", "Case 2", "Case 3"]):
            offset = (i - 1) * width
            vals = []
            for model in ordered_models:
                vals.append(float(sub[(sub["case"] == case) & (sub["model"] == model)]["r2"].iloc[0]))
            ax.bar(
                x + offset,
                vals,
                width=width,
                color=colors[case],
                edgecolor="white",
                linewidth=0.8,
                label=case,
            )

        ax.set_ylabel(r"$R^2$")
        ax.grid(axis="y", alpha=0.25, linewidth=0.8)
        if subset_name == "High amplitude":
            ax.set_ylim(0.75, 0.86)
        else:
            ax.set_ylim(0.69, 0.85)
        ax.legend(loc="upper left", frameon=True, framealpha=0.88, title=subset_note[subset_name], ncol=3)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels([model_to_label[model] for model in ordered_models])
    axes[-1].set_xlabel("Method")

    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_PDF}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()

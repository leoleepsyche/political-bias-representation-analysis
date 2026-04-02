"""
CSV output and plotting helpers for ideology cosine experiments.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .cosine_metrics import (
        MAIN_METHOD_ORDER,
        MATCHED_METHOD_NAME,
        METHOD_ORDER,
        PAIRWISE_METHOD_NAME,
        RANDOM_METHOD_NAME,
        summarize_method,
    )
except ImportError:
    from cosine_metrics import (
        MAIN_METHOD_ORDER,
        MATCHED_METHOD_NAME,
        METHOD_ORDER,
        PAIRWISE_METHOD_NAME,
        RANDOM_METHOD_NAME,
        summarize_method,
    )


PAIR_COLORS = {
    "L-L": "tab:blue",
    "R-R": "tab:red",
    "L-R": "tab:green",
}

TEMPLATE_LABELS = {
    "agree": "agree",
    "agree_yesno": "agree + yes/no",
}


def write_long_csv(path: Path, model_name: str, template: str, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    layer_count = result["layer_count"]
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "template",
                "method",
                "pair_type",
                "layer_id",
                "cosine_mean",
                "cosine_std",
                "num_pairs",
            ],
        )
        if not file_exists:
            writer.writeheader()
        for pair_type, pair_data in result["pair_results"].items():
            for layer_idx in range(layer_count):
                writer.writerow(
                    {
                        "model_name": model_name,
                        "template": template,
                        "method": result["method"],
                        "pair_type": pair_type,
                        "layer_id": layer_idx,
                        "cosine_mean": float(pair_data["mean"][layer_idx]),
                        "cosine_std": float(pair_data["std"][layer_idx]),
                        "num_pairs": int(pair_data["num_pairs"]),
                    }
                )


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "template",
                "method",
                "layer_count",
                "max_separation_layer",
                "min_lr_cosine",
                "mean_lr_cosine",
                "peak_gap_layer",
                "peak_gap_degrees",
                "mean_gap_degrees",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_metadata(
    path: Path,
    input_csv: Path,
    templates: list[str],
    model_name: str,
    left_rows: list[dict],
    right_rows: list[dict],
    random_rounds: int,
    seed: int,
) -> None:
    counts_by_topic = defaultdict(int)
    for row in left_rows + right_rows:
        counts_by_topic[row["topic"]] += 1

    payload = {
        "input_csv": str(input_csv),
        "model_name": model_name,
        "templates": templates,
        "left_count": len(left_rows),
        "right_count": len(right_rows),
        "counts_by_topic": dict(counts_by_topic),
        "random_rounds": random_rounds,
        "seed": seed,
        "methods": METHOD_ORDER,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def plot_single_template_comparison(
    output_path: Path,
    model_name: str,
    template: str,
    template_results: dict[str, dict],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(MAIN_METHOD_ORDER), figsize=(6 * len(MAIN_METHOD_ORDER), 5.8), sharey=True)
    model_short = model_name.split("/")[-1]
    template_label = TEMPLATE_LABELS.get(template, template)

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for axis, method in zip(axes, MAIN_METHOD_ORDER):
        method_title = method
        selector_summary = summarize_method(template_results[method])
        if method == MATCHED_METHOD_NAME:
            method_title = f"{method} (paired L-R only)"
        axis.set_title(method_title, fontsize=14, fontweight="bold")
        axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
        axis.set_xlabel("Layer ID", fontsize=12)

        if method == MATCHED_METHOD_NAME:
            pair_data = template_results[method]["pair_results"]["L-R"]
            layer_ids = np.arange(len(pair_data["mean"]))
            axis.plot(
                layer_ids,
                pair_data["mean"],
                color=PAIR_COLORS["L-R"],
                linestyle="-",
                linewidth=2.4,
                label="L-R",
            )
        else:
            for pair_type in ["L-L", "R-R", "L-R"]:
                pair_data = template_results[method]["pair_results"][pair_type]
                layer_ids = np.arange(len(pair_data["mean"]))
                axis.plot(
                    layer_ids,
                    pair_data["mean"],
                    color=PAIR_COLORS[pair_type],
                    linestyle="-",
                    linewidth=2.0,
                    label=pair_type,
                )

        if method in {RANDOM_METHOD_NAME, PAIRWISE_METHOD_NAME}:
            selector_layer = selector_summary["peak_gap_layer"]
            selector_label = "peak gap"
        else:
            selector_layer = selector_summary["max_separation_layer"]
            selector_label = "max sep"
        if selector_layer is not None:
            axis.axvline(selector_layer, color="black", linestyle=":", linewidth=1.8, alpha=0.85)
            axis.text(
                selector_layer + 0.2,
                0.98,
                selector_label,
                transform=axis.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="left",
                fontsize=9,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
            )

        axis.tick_params(axis="both", labelsize=11)

    axes[0].set_ylabel("Cosine similarity", fontsize=12)
    fig.suptitle(
        f"Ideology cosine comparison · {model_short} · {template_label}\nmain selectors: peak gap for group-level methods, max separation for matched-pair",
        fontsize=16,
        fontweight="bold",
    )
    handles, labels = [], []
    seen_labels = set()
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        for handle, label in zip(axis_handles, axis_labels):
            if label not in seen_labels:
                handles.append(handle)
                labels.append(label)
                seen_labels.add(label)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10, frameon=True)
    fig.tight_layout(rect=[0.02, 0.08, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_group_aggregated_summary(
    output_path: Path,
    model_name: str,
    template: str,
    aggregated_result: dict,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(1, 1, figsize=(6.2, 5.2))
    model_short = model_name.split("/")[-1]
    template_label = TEMPLATE_LABELS.get(template, template)
    pair_data = aggregated_result["pair_results"]["L-R"]
    layer_ids = np.arange(len(pair_data["mean"]))
    summary = summarize_method(aggregated_result)

    axis.plot(
        layer_ids,
        pair_data["mean"],
        color=PAIR_COLORS["L-R"],
        linestyle="-",
        linewidth=2.4,
        label="centroid L-R",
    )
    selector_layer = summary["max_separation_layer"]
    if selector_layer is not None:
        axis.axvline(selector_layer, color="black", linestyle=":", linewidth=1.8, alpha=0.85)
        axis.text(
            selector_layer + 0.2,
            0.98,
            "max sep",
            transform=axis.get_xaxis_transform(),
            rotation=90,
            va="top",
            ha="left",
            fontsize=9,
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
        )

    axis.set_title("group-aggregated (appendix centroid summary)", fontsize=14, fontweight="bold")
    axis.set_xlabel("Layer ID", fontsize=12)
    axis.set_ylabel("Cosine similarity", fontsize=12)
    axis.grid(True, linestyle="--", linewidth=0.7, alpha=0.5)
    axis.tick_params(axis="both", labelsize=11)
    fig.suptitle(
        f"Ideology centroid summary · {model_short} · {template_label}",
        fontsize=15,
        fontweight="bold",
    )
    fig.legend(loc="lower center", ncol=1, fontsize=10, frameon=True)
    fig.tight_layout(rect=[0.02, 0.08, 1, 0.95])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

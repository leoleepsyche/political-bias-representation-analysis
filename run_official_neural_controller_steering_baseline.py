"""
Run official NeuralController steering on IdeoINST left/right data without cosine guidance.

This follows the official repo structure more closely:
- train a controller on all hidden layers from binary left/right detection data
- steer generation on a fixed official-style late-layer range

Local additions are limited to:
- loading IdeoINST matched pairs
- balanced train/val/test splitting
- runtime device compatibility for MPS/CPU
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from run_official_neural_controller_detection import (  # noqa: E402
    _infer_device,
    _patch_cuda_runtime,
    build_official_detection_split,
    load_model_and_tokenizer,
    split_pairs_by_topic,
    summarize_metrics,
    to_builtin,
    WORKSPACE,
)
from run_official_neural_controller_steering import (  # noqa: E402
    run_preview_generations,
)
from neural_controllers import NeuralController  # noqa: E402
import direction_utils as official_direction_utils  # noqa: E402
from step1_dataset import build_paired_rows, load_rows  # noqa: E402


DEFAULT_INPUT = WORKSPACE / "data" / "ideoinst_clean" / "ideoinst_clean_rows.csv"
DEFAULT_OUTPUT = WORKSPACE / "outputs" / "official_nc_steering_baseline"


def build_held_out_pairs(test_pairs: list[tuple[dict, dict]]) -> list[dict]:
    return [
        {
            "instruction_id": left_row["instruction_id"],
            "topic": left_row["topic"],
            "instruction": left_row["instruction"],
            "left_response": left_row["response_text"],
            "right_response": right_row["response_text"],
        }
        for left_row, right_row in test_pairs
    ]


def official_range_layers(hidden_layers: list[int], start_negative: int) -> list[int]:
    selected = [layer for layer in hidden_layers if layer <= start_negative]
    if not selected:
        raise ValueError(
            f"No hidden layers selected for start_negative={start_negative}. "
            f"Available layers: {hidden_layers}"
        )
    return selected


def build_detection_prompts(controller: NeuralController, texts: list[str]) -> list[str]:
    prompts: list[str] = []
    for text in texts:
        prompt = (
            "Which political party supports the following position?\n"
            f"Position: {text}"
        )
        prompts.append(controller.format_prompt(prompt))
    return prompts


def score_previews_with_detector(
    controller: NeuralController,
    previews: list[dict],
    rep_layer: int,
) -> tuple[list[dict], dict]:
    condition_texts: dict[str, list[str]] = defaultdict(list)
    for preview in previews:
        condition_texts["no_control"].append(preview["no_control"])
        for coef_key, text in preview["steered"].items():
            condition_texts[coef_key].append(text)

    condition_scores: dict[str, list[float]] = {}
    for condition, texts in condition_texts.items():
        prompts = build_detection_prompts(controller, texts)
        hidden_states = official_direction_utils.get_hidden_states(
            prompts,
            controller.model,
            controller.tokenizer,
            controller.hidden_layers,
            controller.hyperparams["forward_batch_size"],
            all_positions=True,
        )
        projections = official_direction_utils.project_hidden_states(
            hidden_states,
            controller.directions,
            controller.hyperparams["n_components"],
        )
        x = projections[rep_layer]
        beta, b = controller.detector_coefs[rep_layer]
        if hasattr(beta, "to"):
            beta = beta.to(device=x.device, dtype=x.dtype)
        if hasattr(b, "to"):
            b = b.to(device=x.device, dtype=x.dtype)
        scores = x @ beta + b
        if hasattr(scores, "detach"):
            scores = scores.detach().reshape(-1).cpu().tolist()
        elif not isinstance(scores, list):
            scores = [float(scores)]
        else:
            scores = torch.tensor(scores).reshape(-1).tolist()
        condition_scores[condition] = [float(score) for score in scores]

    scored_previews: list[dict] = []
    for idx, preview in enumerate(previews):
        scored_preview = dict(preview)
        scored_preview["detect_scores"] = {
            "no_control": condition_scores["no_control"][idx],
            "steered": {
                coef_key: condition_scores[coef_key][idx]
                for coef_key in preview["steered"].keys()
            },
        }
        scored_previews.append(scored_preview)

    no_control_scores = condition_scores["no_control"]
    summary = {
        "no_control": {
            "mean_detect_score": sum(no_control_scores) / len(no_control_scores),
            "score_gt_0_5_rate": sum(score > 0.5 for score in no_control_scores) / len(no_control_scores),
        }
    }
    for condition, scores in condition_scores.items():
        if condition == "no_control":
            continue
        summary[condition] = {
            "mean_detect_score": sum(scores) / len(scores),
            "score_gt_0_5_rate": sum(score > 0.5 for score in scores) / len(scores),
            "delta_vs_no_control": (sum(scores) / len(scores)) - summary["no_control"]["mean_detect_score"],
        }

    summary["rep_layer"] = rep_layer

    return scored_previews, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official NeuralController steering baseline on IdeoINST left/right data."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-method", choices=["rfm", "pca", "mean_difference"], default="rfm")
    parser.add_argument("--target-ideology", choices=["left", "right"], default="left")
    parser.add_argument("--train-pairs", type=int, default=60)
    parser.add_argument("--val-pairs", type=int, default=30)
    parser.add_argument("--test-pairs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-components", type=int, default=1)
    parser.add_argument("--rfm-iters", type=int, default=8)
    parser.add_argument(
        "--official-range-start-negative",
        type=int,
        default=-8,
        help="Control all layers from this negative index downward, mirroring the paper notebook.",
    )
    parser.add_argument("--control-coefs", nargs="*", type=float, default=[0.8, 1.5, 2.5, 4.0])
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="test",
        help="Which held-out split to steer and score. Use val for coefficient selection and test for final evaluation.",
    )
    parser.add_argument("--normalize-total-strength", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _infer_device(args.device)
    compute_device = "cpu" if args.control_method == "rfm" and device != "cuda" else device
    _patch_cuda_runtime(device, rfm_device=compute_device)

    rows = load_rows(args.input_csv)
    paired_rows = build_paired_rows(rows, strict=True)
    train_pairs, val_pairs, test_pairs = split_pairs_by_topic(
        paired_rows,
        train_pairs=args.train_pairs,
        val_pairs=args.val_pairs,
        test_pairs=args.test_pairs,
        seed=args.seed,
    )

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    controller = NeuralController(
        model,
        tokenizer,
        rfm_iters=args.rfm_iters,
        batch_size=args.batch_size,
        n_components=args.n_components,
        control_method=args.control_method,
    )

    train_inputs, train_labels = build_official_detection_split(
        controller, train_pairs, positive_ideology=args.target_ideology
    )
    val_inputs, val_labels = build_official_detection_split(
        controller, val_pairs, positive_ideology=args.target_ideology
    )
    test_inputs, test_labels = build_official_detection_split(
        controller, test_pairs, positive_ideology=args.target_ideology
    )

    controller.compute_directions(
        train_inputs,
        train_labels,
        val_data=val_inputs,
        val_labels=val_labels,
        hidden_layers=controller.hidden_layers,
        device=compute_device,
    )

    val_metrics, test_metrics, detector_coefs, _ = controller.evaluate_directions(
        train_inputs,
        train_labels,
        val_inputs,
        val_labels,
        test_inputs,
        test_labels,
        hidden_layers=controller.hidden_layers,
        agg_model=args.control_method,
        selection_metric="auc",
    )
    controller.detector_coefs = detector_coefs
    detector_summary = summarize_metrics(val_metrics, test_metrics, "auc")
    rep_layer = int(detector_summary["best_layer_on_val"])

    selected_layers = official_range_layers(
        controller.hidden_layers,
        start_negative=args.official_range_start_negative,
    )
    eval_pairs = val_pairs if args.eval_split == "val" else test_pairs
    previews = run_preview_generations(
        controller=controller,
        test_pairs=build_held_out_pairs(eval_pairs),
        selected_layers=selected_layers,
        control_coefs=args.control_coefs,
        max_new_tokens=args.max_new_tokens,
        preview_count=args.preview_count,
        normalize_total_strength=args.normalize_total_strength,
    )
    scored_previews, detect_summary = score_previews_with_detector(controller, previews, rep_layer=rep_layer)

    effective_control_coefs = [
        coef / math.sqrt(len(selected_layers))
        if args.normalize_total_strength and selected_layers
        else coef
        for coef in args.control_coefs
    ]

    output_dir = args.output_dir / args.target_ideology
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model": args.model,
        "device": device,
        "compute_device": compute_device,
        "control_method": args.control_method,
        "target_ideology": args.target_ideology,
        "train_pairs": args.train_pairs,
        "val_pairs": args.val_pairs,
        "test_pairs": args.test_pairs,
        "official_range_start_negative": args.official_range_start_negative,
        "selected_layers_negative_index": selected_layers,
        "control_coefs": args.control_coefs,
        "effective_control_coefs": effective_control_coefs,
        "normalize_total_strength": args.normalize_total_strength,
        "preview_count": args.preview_count,
        "eval_split": args.eval_split,
        "eval_pair_count": len(eval_pairs) if args.preview_count <= 0 else min(args.preview_count, len(eval_pairs)),
        "max_new_tokens": args.max_new_tokens,
        "input_csv": str(args.input_csv),
        "detector_summary": detector_summary,
        "steering_detect_summary": detect_summary,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(to_builtin(metadata), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "detector_summary.json").write_text(
        json.dumps(to_builtin(detector_summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "steering_detect_summary.json").write_text(
        json.dumps(to_builtin(detect_summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "preview_generations.json").write_text(
        json.dumps(to_builtin(scored_previews), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(to_builtin(metadata), ensure_ascii=False, indent=2))
    print(f"Saved preview outputs to: {output_dir / 'preview_generations.json'}")


if __name__ == "__main__":
    main()

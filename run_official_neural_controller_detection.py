"""
Run the official NeuralController detection pipeline on IdeoINST left/right data.

This script keeps the official controller code path intact:
- format prompts with the official controller
- compute directions with `NeuralController.compute_directions`
- evaluate with `NeuralController.evaluate_directions`

The only local additions are:
- loading matched left/right rows from ideoinst_clean_rows.csv
- balanced train/val/test splitting by topic
- a thin runtime compatibility shim so the CUDA-first code can run on MPS/CPU
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_INPUT = WORKSPACE / "data" / "ideoinst_clean" / "ideoinst_clean_rows.csv"
DEFAULT_OUTPUT = WORKSPACE / "outputs" / "official_neural_controller_detection"
OFFICIAL_REPO = WORKSPACE.parent / "neural_controllers_official"

if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from neural_controllers import NeuralController  # noqa: E402
import direction_utils as official_direction_utils  # noqa: E402
from step1_dataset import (  # noqa: E402
    allocate_topic_counts,
    build_paired_rows,
    load_rows,
    ordered_topics,
)


def _infer_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _patch_cuda_runtime(default_device: str, rfm_device: str | None = None) -> None:
    """Make the official CUDA-first code path usable on MPS/CPU without edits."""
    if default_device == "cuda":
        return

    target = torch.device(default_device)
    rfm_target = rfm_device or default_device
    original_aggregate_layers = official_direction_utils.aggregate_layers
    original_get_hidden_states = official_direction_utils.get_hidden_states
    original_linear_solve = official_direction_utils.linear_solve
    original_compute_prediction_metrics = official_direction_utils.compute_prediction_metrics

    def _clean_tensor(values, *, nan: float = 0.0, posinf: float = 0.0, neginf: float = 0.0):
        if isinstance(values, torch.Tensor) and torch.is_floating_point(values):
            return torch.nan_to_num(values, nan=nan, posinf=posinf, neginf=neginf)
        return values

    def _tensor_cuda(self, device=None, non_blocking=False, memory_format=None):
        del device, memory_format
        return self.to(device=target, non_blocking=non_blocking)

    def _module_cuda(self, device=None):
        del device
        return self.to(device=target)

    def _project_onto_direction(tensors, direction, device="cuda"):
        del device
        assert len(tensors.shape) == 2
        assert tensors.shape[1] == direction.shape[0]
        return tensors.to(device=target) @ direction.to(device=target, dtype=tensors.dtype)

    def _project_hidden_states(hidden_states, directions, n_components):
        assert hidden_states.keys() == directions.keys()
        projections = {}
        for layer in hidden_states.keys():
            layer_hidden = hidden_states[layer].to(device=target)
            if not torch.isfinite(layer_hidden).all():
                layer_hidden = torch.nan_to_num(layer_hidden, nan=0.0, posinf=0.0, neginf=0.0)
            vecs = directions[layer][:n_components].T
            if hasattr(vecs, "to"):
                vecs = vecs.to(device=target, dtype=layer_hidden.dtype)
            projections[layer] = layer_hidden @ vecs
        return projections

    def _get_hidden_states(prompts, model, tokenizer, hidden_layers, forward_batch_size, rep_token=-1, all_positions=False):
        hidden_states = original_get_hidden_states(
            prompts,
            model,
            tokenizer,
            hidden_layers,
            forward_batch_size,
            rep_token=rep_token,
            all_positions=all_positions,
        )
        cleaned_hidden_states = {}
        for layer, states in hidden_states.items():
            if torch.isfinite(states).all():
                cleaned_hidden_states[layer] = states
            else:
                cleaned_hidden_states[layer] = torch.nan_to_num(
                    states, nan=0.0, posinf=0.0, neginf=0.0
                )
        return cleaned_hidden_states

    def _linear_solve(X, y, use_bias=True, reg=0):
        X = _clean_tensor(X.float())
        y = _clean_tensor(y.float())
        beta, bias = original_linear_solve(X, y, use_bias=use_bias, reg=reg)
        beta = _clean_tensor(beta)
        if isinstance(bias, torch.Tensor):
            bias = _clean_tensor(bias)
        elif isinstance(bias, float) and not math.isfinite(bias):
            bias = 0.0
        return beta, bias

    def _compute_prediction_metrics(preds, labels, classification_threshold=0.5):
        preds = _clean_tensor(preds, nan=0.5, posinf=1.0, neginf=0.0)
        labels = _clean_tensor(labels, nan=0.0, posinf=1.0, neginf=0.0)
        return original_compute_prediction_metrics(
            preds,
            labels,
            classification_threshold=classification_threshold,
        )

    def _aggregate_projections_on_coefs(projections, detector_coef):
        agg_projections = []
        for layer in projections.keys():
            X = projections[layer].to(device=target)
            agg_projections.append(X.squeeze(0))

        agg_projections = torch.concat(agg_projections, dim=1).squeeze()
        agg_beta = detector_coef[0]
        agg_bias = detector_coef[1]
        if hasattr(agg_beta, "to"):
            agg_beta = agg_beta.to(device=target, dtype=agg_projections.dtype)
        if hasattr(agg_bias, "to"):
            agg_bias = agg_bias.to(device=target, dtype=agg_projections.dtype)
        agg_preds = agg_projections @ agg_beta + agg_bias
        return agg_preds

    def _train_rfm_probe_on_concept(
        train_X,
        train_y,
        val_X,
        val_y,
        hyperparams,
        search_space=None,
        tuning_metric="auc",
    ):
        if search_space is None:
            search_space = {
                "regs": [1e-3],
                "bws": [1, 10, 100],
                "center_grads": [True, False],
            }

        train_X_local = _clean_tensor(train_X.to(rfm_target).float())
        train_y_local = _clean_tensor(train_y.to(rfm_target).float())
        val_X_local = _clean_tensor(val_X.to(rfm_target).float())
        val_y_local = _clean_tensor(val_y.to(rfm_target).float())

        best_model = None
        maximize_metric = tuning_metric in ["f1", "auc", "acc", "top_agop_vectors_ols_auc"]
        best_score = float("-inf") if maximize_metric else float("inf")
        best_reg = None
        best_bw = None
        best_center_grads = None

        for reg in search_space["regs"]:
            for bw in search_space["bws"]:
                for center_grads in search_space["center_grads"]:
                    try:
                        rfm_params = {
                            "model": {
                                "kernel": "l2_high_dim",
                                "bandwidth": bw,
                                "tuning_metric": tuning_metric,
                            },
                            "fit": {
                                "reg": reg,
                                "iters": hyperparams["rfm_iters"],
                                "center_grads": center_grads,
                                "early_stop_rfm": True,
                                "get_agop_best_model": True,
                                "top_k": hyperparams["n_components"],
                            },
                        }
                        model = official_direction_utils.RFM(
                            **rfm_params["model"], device=rfm_target
                        )
                        model.fit(
                            (train_X_local, train_y_local),
                            (val_X_local, val_y_local),
                            **rfm_params["fit"],
                        )

                        if tuning_metric == "top_agop_vectors_ols_auc":
                            top_k = hyperparams["n_components"]
                            targets = val_y_local
                            _, U = torch.lobpcg(model.agop_best_model, k=top_k)
                            top_eigenvectors = U[:, :top_k]
                            projections = val_X_local @ top_eigenvectors
                            projections = projections.reshape(-1, top_k)
                            xtx = projections.T @ projections
                            xty = projections.T @ targets
                            betas = torch.linalg.pinv(xtx) @ xty
                            preds = torch.sigmoid(projections @ betas).reshape(targets.shape)
                            preds = _clean_tensor(preds, nan=0.5, posinf=1.0, neginf=0.0)
                            val_score = official_direction_utils.roc_auc_score(
                                targets.cpu().numpy(),
                                preds.cpu().numpy(),
                            )
                        else:
                            pred_proba = model.predict(val_X_local)
                            pred_proba = _clean_tensor(pred_proba, nan=0.5, posinf=1.0, neginf=0.0)
                            val_score = official_direction_utils.compute_prediction_metrics(
                                pred_proba,
                                val_y_local,
                            )[tuning_metric]

                        if (maximize_metric and val_score > best_score) or (
                            not maximize_metric and val_score < best_score
                        ):
                            best_score = val_score
                            best_reg = reg
                            best_bw = bw
                            best_center_grads = center_grads
                            best_model = official_direction_utils.deepcopy(model)
                    except Exception as exc:  # pragma: no cover - passthrough logging
                        import traceback

                        print(f"Error fitting RFM on device={rfm_target}: {traceback.format_exc()}")
                        print(f"Original exception: {exc}")
                        continue

        print(
            f"Best RFM {tuning_metric}: {best_score}, reg: {best_reg}, "
            f"bw: {best_bw}, center_grads: {best_center_grads}, device: {rfm_target}"
        )
        return best_model

    def _aggregate_layers(layer_outputs, train_y, val_y, test_y, agg_model="linear", tuning_metric="auc"):
        if agg_model != "rfm":
            return original_aggregate_layers(
                layer_outputs,
                train_y,
                val_y,
                test_y,
                agg_model=agg_model,
                tuning_metric=tuning_metric,
            )

        train_X = _clean_tensor(torch.concat(layer_outputs["train"], dim=1).to(rfm_target).float())
        val_X = _clean_tensor(torch.concat(layer_outputs["val"], dim=1).to(rfm_target).float())
        test_X = _clean_tensor(torch.concat(layer_outputs["test"], dim=1).to(rfm_target).float())
        train_y = _clean_tensor(train_y.to(rfm_target).float())
        val_y = _clean_tensor(val_y.to(rfm_target).float())
        test_y = _clean_tensor(test_y.to(rfm_target).float())

        bw_search_space = [10]
        reg_search_space = [1e-4, 1e-3, 1e-2]
        kernel_search_space = ["l2_high_dim"]
        maximize_metric = tuning_metric in ["f1", "auc", "acc"]

        best_rfm_params = None
        best_rfm_score = float("-inf") if maximize_metric else float("inf")
        for bw in bw_search_space:
            for reg in reg_search_space:
                for kernel in kernel_search_space:
                    rfm_params = {
                        "model": {
                            "kernel": kernel,
                            "bandwidth": bw,
                        },
                        "fit": {
                            "reg": reg,
                            "iters": 10,
                        },
                    }
                    model = official_direction_utils.xRFM(
                        rfm_params, device=rfm_target, tuning_metric=tuning_metric
                    )
                    model.fit(train_X, train_y, val_X, val_y)
                    val_preds = _clean_tensor(model.predict(val_X), nan=0.5, posinf=1.0, neginf=0.0)
                    metrics = official_direction_utils.compute_prediction_metrics(val_preds, val_y)

                    if (maximize_metric and metrics[tuning_metric] > best_rfm_score) or (
                        not maximize_metric and metrics[tuning_metric] < best_rfm_score
                    ):
                        best_rfm_score = metrics[tuning_metric]
                        best_rfm_params = rfm_params

        model = official_direction_utils.xRFM(best_rfm_params, device=rfm_target, tuning_metric=tuning_metric)
        model.fit(train_X, train_y, val_X, val_y)
        test_preds = _clean_tensor(model.predict(test_X), nan=0.5, posinf=1.0, neginf=0.0)
        metrics = official_direction_utils.compute_prediction_metrics(test_preds, test_y)
        return metrics, None, None, test_preds

    torch.Tensor.cuda = _tensor_cuda  # type: ignore[assignment]
    torch.nn.Module.cuda = _module_cuda  # type: ignore[assignment]
    official_direction_utils.project_onto_direction = _project_onto_direction
    official_direction_utils.project_hidden_states = _project_hidden_states
    official_direction_utils.get_hidden_states = _get_hidden_states
    official_direction_utils.aggregate_projections_on_coefs = _aggregate_projections_on_coefs
    official_direction_utils.linear_solve = _linear_solve
    official_direction_utils.compute_prediction_metrics = _compute_prediction_metrics
    official_direction_utils.train_rfm_probe_on_concept = _train_rfm_probe_on_concept
    official_direction_utils.aggregate_layers = _aggregate_layers


def load_model_and_tokenizer(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
        )
        model.to(device)

    model.eval()
    return model, tokenizer


def split_pairs_by_topic(
    paired_rows: list[tuple[dict, dict]],
    train_pairs: int,
    val_pairs: int,
    test_pairs: int,
    seed: int,
) -> tuple[list[tuple[dict, dict]], list[tuple[dict, dict]], list[tuple[dict, dict]]]:
    paired_left_rows = [left_row for left_row, _ in paired_rows]
    train_counts = allocate_topic_counts(paired_left_rows, train_pairs)
    val_counts = allocate_topic_counts(paired_left_rows, val_pairs)
    test_counts = allocate_topic_counts(paired_left_rows, test_pairs)

    grouped_pairs: dict[str, list[tuple[dict, dict]]] = {}
    for left_row, right_row in paired_rows:
        grouped_pairs.setdefault(left_row["topic"], []).append((left_row, right_row))

    train_split: list[tuple[dict, dict]] = []
    val_split: list[tuple[dict, dict]] = []
    test_split: list[tuple[dict, dict]] = []

    rng = random.Random(seed)
    for topic in ordered_topics(paired_left_rows):
        topic_pairs = list(grouped_pairs[topic])
        rng.shuffle(topic_pairs)

        train_n = train_counts.get(topic, 0)
        val_n = val_counts.get(topic, 0)
        test_n = test_counts.get(topic, 0)
        total_needed = train_n + val_n + test_n
        if len(topic_pairs) < total_needed:
            raise ValueError(
                f"Topic {topic} has {len(topic_pairs)} matched pairs, "
                f"cannot allocate train={train_n}, val={val_n}, test={test_n}."
            )

        train_split.extend(topic_pairs[:train_n])
        val_split.extend(topic_pairs[train_n : train_n + val_n])
        test_split.extend(topic_pairs[train_n + val_n : total_needed])

    return train_split, val_split, test_split


def build_official_detection_split(
    controller: NeuralController,
    paired_rows: list[tuple[dict, dict]],
    positive_ideology: str,
) -> tuple[list[str], list[float]]:
    if positive_ideology not in {"left", "right"}:
        raise ValueError("positive_ideology must be 'left' or 'right'.")

    inputs: list[str] = []
    labels: list[float] = []

    for left_row, right_row in paired_rows:
        ideology_to_row = {"left": left_row, "right": right_row}
        positive_row = ideology_to_row[positive_ideology]
        negative_row = ideology_to_row["right" if positive_ideology == "left" else "left"]

        for row, label in ((positive_row, 1.0), (negative_row, 0.0)):
            prompt = (
                "Which political party supports the following position?\n"
                f"Position: {row['response_text']}"
            )
            inputs.append(controller.format_prompt(prompt))
            labels.append(label)

    return inputs, labels


def summarize_metrics(val_metrics: dict, test_metrics: dict, selection_metric: str) -> dict:
    best_layer = max(
        (layer for layer in val_metrics.keys() if isinstance(layer, int)),
        key=lambda layer: val_metrics[layer][selection_metric],
    )

    return {
        "selection_metric": selection_metric,
        "best_layer_on_val": int(best_layer),
        "best_layer_val_metrics": val_metrics[best_layer],
        "best_layer_test_metrics": test_metrics[best_layer],
        "official_best_layer_test_metrics": test_metrics.get("best_layer"),
        "aggregation_test_metrics": test_metrics.get("aggregation"),
    }


def to_builtin(value):
    if isinstance(value, dict):
        return {str(key): to_builtin(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    return value


def write_metrics_csv(output_path: Path, split_name: str, metrics: dict) -> None:
    rows: list[dict] = []
    for layer_key, layer_metrics in metrics.items():
        layer_name = str(layer_key)
        for metric_name, metric_value in layer_metrics.items():
            rows.append(
                {
                    "split": split_name,
                    "layer": layer_name,
                    "metric": metric_name,
                    "value": float(metric_value),
                }
            )

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "layer", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run official NeuralController detection on IdeoINST left/right data."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--control-method", default="pca")
    parser.add_argument("--positive-ideology", choices=["left", "right"], default="left")
    parser.add_argument("--train-pairs", type=int, default=100)
    parser.add_argument("--val-pairs", type=int, default=50)
    parser.add_argument("--test-pairs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--n-components", type=int, default=1)
    parser.add_argument("--rfm-iters", type=int, default=8)
    parser.add_argument("--selection-metric", default="auc")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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
        controller, train_pairs, positive_ideology=args.positive_ideology
    )
    val_inputs, val_labels = build_official_detection_split(
        controller, val_pairs, positive_ideology=args.positive_ideology
    )
    test_inputs, test_labels = build_official_detection_split(
        controller, test_pairs, positive_ideology=args.positive_ideology
    )

    hidden_layers = controller.hidden_layers
    train_hidden = official_direction_utils.get_hidden_states(
        train_inputs, model, tokenizer, hidden_layers, args.batch_size
    )
    val_hidden = official_direction_utils.get_hidden_states(
        val_inputs, model, tokenizer, hidden_layers, args.batch_size
    )
    test_hidden = official_direction_utils.get_hidden_states(
        test_inputs, model, tokenizer, hidden_layers, args.batch_size
    )

    controller.compute_directions(
        train_hidden,
        train_labels,
        val_hidden,
        val_labels,
        device=compute_device,
    )
    agg_model = args.control_method if args.control_method in {"rfm", "linear", "logistic"} else "linear"
    val_metrics, test_metrics, detector_coefs, _ = controller.evaluate_directions(
        train_hidden,
        train_labels,
        val_hidden,
        val_labels,
        test_hidden,
        test_labels,
        agg_model=agg_model,
        selection_metric=args.selection_metric,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = to_builtin(summarize_metrics(val_metrics, test_metrics, args.selection_metric))
    metadata = {
        "model": args.model,
        "device": device,
        "compute_device": compute_device,
        "control_method": args.control_method,
        "positive_ideology": args.positive_ideology,
        "train_pairs": args.train_pairs,
        "val_pairs": args.val_pairs,
        "test_pairs": args.test_pairs,
        "train_inputs": len(train_inputs),
        "val_inputs": len(val_inputs),
        "test_inputs": len(test_inputs),
        "selection_metric": args.selection_metric,
    }
    metadata = to_builtin(metadata)

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with (output_dir / "detector_layers.json").open("w", encoding="utf-8") as handle:
        json.dump(
            sorted(int(layer) for layer in detector_coefs.keys() if isinstance(layer, int)),
            handle,
            indent=2,
        )

    write_metrics_csv(output_dir / "val_metrics_long.csv", "val", val_metrics)
    write_metrics_csv(output_dir / "test_metrics_long.csv", "test", test_metrics)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

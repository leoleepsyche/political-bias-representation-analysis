"""
Run single-layer ideology steering with the paper's official NeuralController code.

This adapter keeps the official controller implementation intact and only adds:

- local dataset preparation from ideoinst_clean_rows.csv
- selected-layer lookup from an existing cosine summary CSV
- a thin device compatibility patch for non-CUDA execution
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_INPUT = WORKSPACE / "data" / "ideoinst_clean" / "ideoinst_clean_rows.csv"
DEFAULT_SUMMARY = (
    WORKSPACE
    / "outputs"
    / "ideology_cosine_clean50x6_qwen7b_instruct_agree"
    / "ideology_cosine_summary.csv"
)
DEFAULT_OUTPUT = WORKSPACE / "outputs" / "official_neural_controller_steering"
OFFICIAL_REPO = WORKSPACE.parent / "neural_controllers_official"

if str(OFFICIAL_REPO) not in sys.path:
    sys.path.insert(0, str(OFFICIAL_REPO))
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from neural_controllers import NeuralController  # noqa: E402
import direction_utils as official_direction_utils  # noqa: E402
from step1_dataset import load_rows, prepare_rows  # noqa: E402
from run_official_neural_controller_detection import _patch_cuda_runtime  # noqa: E402


def _patch_official_device_support(default_device: str) -> None:
    """The official repo is CUDA-first; this keeps the same logic on MPS/CPU."""

    def _project_onto_direction(tensors, direction, device: str = default_device):
        assert len(tensors.shape) == 2
        assert tensors.shape[1] == direction.shape[0]
        return tensors.to(device=device) @ direction.to(device=device, dtype=tensors.dtype)

    def _get_hidden_states(
        prompts,
        model,
        tokenizer,
        hidden_layers,
        forward_batch_size,
        rep_token: int = -1,
        all_positions: bool = False,
    ):
        if isinstance(prompts, tuple):
            prompts = list(prompts)

        encoded_inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(model.device)
        encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"].half()

        dataset = TensorDataset(encoded_inputs["input_ids"], encoded_inputs["attention_mask"])
        dataloader = DataLoader(dataset, batch_size=forward_batch_size)

        all_hidden_states = {layer_idx: [] for layer_idx in hidden_layers}

        with torch.no_grad():
            for batch in dataloader:
                input_ids, attention_mask = batch
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                num_layers = len(model.model.layers)
                out_hidden_states = outputs.hidden_states

                # `outputs.hidden_states` includes the embedding activations at index 0.
                # We want the actual transformer blocks only, mapping:
                # layer 0 -> -num_layers, ..., top layer -> -1.
                for layer_idx, hidden_state in zip(
                    range(-1, -num_layers - 1, -1),
                    reversed(out_hidden_states[1:]),
                ):
                    if layer_idx not in all_hidden_states:
                        continue
                    hidden_state = torch.nan_to_num(
                        hidden_state.detach().cpu(),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    if all_positions:
                        all_hidden_states[layer_idx].append(hidden_state)
                    else:
                        all_hidden_states[layer_idx].append(
                            hidden_state[:, rep_token, :]
                        )

        return {
            layer_idx: torch.cat(hidden_state_list, dim=0)
            for layer_idx, hidden_state_list in all_hidden_states.items()
        }

    def _fit_pca_model(train_X, train_y, n_components: int = 1, mean_center: bool = True):
        pos_indices = torch.isclose(train_y, torch.ones_like(train_y)).squeeze(1)
        neg_indices = torch.isclose(train_y, torch.zeros_like(train_y)).squeeze(1)

        pos_examples = train_X[pos_indices]
        neg_examples = train_X[neg_indices]
        dif_vectors = pos_examples - neg_examples

        random_signs = (
            torch.randint(0, 2, (len(dif_vectors),), device=dif_vectors.device).float() * 2 - 1
        )
        dif_vectors = dif_vectors * random_signs.reshape(-1, 1)
        if mean_center:
            dif_vectors = dif_vectors - torch.mean(dif_vectors, dim=0, keepdim=True)

        xtx = (dif_vectors.T @ dif_vectors).float().cpu()
        _, eigenvectors = torch.lobpcg(xtx, k=n_components)
        return eigenvectors.T.to(train_X.device)

    official_direction_utils.project_onto_direction = _project_onto_direction
    official_direction_utils.get_hidden_states = _get_hidden_states
    official_direction_utils.fit_pca_model = _fit_pca_model


def _infer_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def candidate_model_names(model_name: str) -> list[str]:
    candidates = [model_name]
    snapshot_match = re.search(r"models--([^/]+)--([^/]+)/snapshots/", model_name)
    if snapshot_match:
        org, model = snapshot_match.groups()
        candidates.append(f"{org}/{model}")
    try:
        name_path = Path(model_name)
        if name_path.name:
            candidates.append(name_path.name)
        if name_path.parent.name.startswith("models--"):
            repo_bits = name_path.parent.name.removeprefix("models--").split("--", maxsplit=1)
            if len(repo_bits) == 2:
                candidates.append(f"{repo_bits[0]}/{repo_bits[1]}")
    except OSError:
        pass

    deduped: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in deduped:
            deduped.append(candidate)
    return deduped


def select_layer_from_summary(
    summary_csv: Path,
    model_name: str,
    template: str,
    method: str,
) -> int:
    with Path(summary_csv).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    model_name_candidates = set(candidate_model_names(model_name))
    for row in rows:
        if (
            row["model_name"] in model_name_candidates
            and row["template"] == template
            and row["method"] == method
        ):
            peak_gap_layer = row.get("peak_gap_layer", "").strip()
            max_sep_layer = row.get("max_separation_layer", "").strip()
            if peak_gap_layer:
                return int(float(peak_gap_layer))
            if max_sep_layer:
                return int(float(max_sep_layer))
            raise ValueError(
                f"Summary row found but contains no selectable layer: {row}"
            )

    raise ValueError(
        "No summary row found for "
        f"model candidates={sorted(model_name_candidates)!r}, template={template!r}, method={method!r}"
    )


def map_layer_to_negative_index(layer_id: int, num_hidden_layers: int) -> int:
    negative_idx = layer_id - num_hidden_layers
    if negative_idx >= 0:
        raise ValueError(
            f"Invalid mapped layer index {negative_idx} from layer_id={layer_id}, "
            f"num_hidden_layers={num_hidden_layers}."
        )
    return negative_idx


def build_local_window_layers(
    peak_layer: int,
    num_hidden_layers: int,
    window_size: int,
) -> list[int]:
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError(f"window_size must be a positive odd integer, got {window_size}")
    radius = window_size // 2
    start = max(0, peak_layer - radius)
    end = min(num_hidden_layers - 1, peak_layer + radius)
    layers = list(range(start, end + 1))
    if len(layers) < window_size:
        if start == 0:
            end = min(num_hidden_layers - 1, window_size - 1)
            layers = list(range(0, end + 1))
        elif end == num_hidden_layers - 1:
            start = max(0, num_hidden_layers - window_size)
            layers = list(range(start, num_hidden_layers))
    return layers


def sample_random_layers(
    num_hidden_layers: int,
    layer_count: int,
    excluded_layers: list[int],
    seed: int,
) -> list[int]:
    candidates = [layer for layer in range(num_hidden_layers) if layer not in set(excluded_layers)]
    if layer_count > len(candidates):
        raise ValueError(
            f"Cannot sample {layer_count} random layers from {len(candidates)} candidates."
        )
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, k=layer_count))


def build_binary_pair_dataset(
    left_rows: list[dict],
    right_rows: list[dict],
    tokenizer,
    target_ideology: str,
    train_pairs: int,
    val_pairs: int,
    test_pairs: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    if target_ideology not in {"left", "right"}:
        raise ValueError(f"Unknown target_ideology {target_ideology!r}")

    paired = list(zip(left_rows, right_rows))
    rng = random.Random(seed)
    rng.shuffle(paired)

    needed_pairs = train_pairs + val_pairs + test_pairs
    if needed_pairs > len(paired):
        raise ValueError(
            f"Requested {needed_pairs} paired examples but only {len(paired)} are available."
        )

    paired = paired[:needed_pairs]
    train_set = paired[:train_pairs]
    val_set = paired[train_pairs:train_pairs + val_pairs]
    test_set = paired[train_pairs + val_pairs:]

    def _format_probe_prompt(row: dict) -> str:
        user_text = (
            "Which political ideology is best represented by the following position?\n"
            f"Position: {row['response_text']}\n"
            "Answer:"
        )
        chat = [{"role": "user", "content": user_text}]
        return tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

    def _pair_to_examples(examples: list[tuple[dict, dict]]) -> tuple[list[str], list[float]]:
        inputs: list[str] = []
        labels: list[float] = []
        for left_row, right_row in examples:
            positive_row = left_row if target_ideology == "left" else right_row
            negative_row = right_row if target_ideology == "left" else left_row
            inputs.extend([
                _format_probe_prompt(positive_row),
                _format_probe_prompt(negative_row),
            ])
            labels.extend([1.0, 0.0])
        return inputs, labels

    train_inputs, train_labels = _pair_to_examples(train_set)
    val_inputs, val_labels = _pair_to_examples(val_set)
    test_inputs, test_labels = _pair_to_examples(test_set)
    return {
        "train_inputs": train_inputs,
        "train_labels": train_labels,
        "val_inputs": val_inputs,
        "val_labels": val_labels,
        "test_inputs": test_inputs,
        "test_labels": test_labels,
    }, [
        {
            "instruction_id": left_row["instruction_id"],
            "topic": left_row["topic"],
            "instruction": left_row["instruction"],
            "left_response": left_row["response_text"],
            "right_response": right_row["response_text"],
        }
        for left_row, right_row in test_set
    ]


def trim_completion(prompt: str, generated_text: str) -> str:
    if generated_text.startswith(prompt):
        return generated_text[len(prompt):].strip()
    return generated_text.strip()


def run_preview_generations(
    controller: NeuralController,
    test_pairs: list[dict],
    selected_layers: list[int],
    control_coefs: list[float],
    max_new_tokens: int,
    preview_count: int,
    normalize_total_strength: bool,
) -> list[dict]:
    previews: list[dict] = []
    layer_count = len(selected_layers)
    pairs_to_run = test_pairs if preview_count <= 0 else test_pairs[:preview_count]
    for pair in pairs_to_run:
        formatted_prompt = controller.format_prompt(pair["instruction"], steer=True)
        no_control = controller.generate(
            formatted_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        steered_outputs = {}
        for control_coef in control_coefs:
            effective_coef = (
                control_coef / math.sqrt(layer_count)
                if normalize_total_strength and layer_count > 0
                else control_coef
            )
            steered = controller.generate(
                formatted_prompt,
                layers_to_control=selected_layers,
                control_coef=effective_coef,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            steered_outputs[f"coef_{control_coef:g}"] = trim_completion(formatted_prompt, steered)
        previews.append(
            {
                "instruction_id": pair["instruction_id"],
                "topic": pair["topic"],
                "instruction": pair["instruction"],
                "no_control": trim_completion(formatted_prompt, no_control),
                "steered": steered_outputs,
            }
        )
    return previews


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run official NeuralController single-layer steering on left/right ideology data."
    )
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--template", type=str, default="agree")
    parser.add_argument("--selection-method", type=str, default="pairwise")
    parser.add_argument(
        "--layer-window-size",
        type=int,
        default=1,
        help="Odd window size centered on the cosine-selected peak layer. Use 1 for single-layer steering.",
    )
    parser.add_argument(
        "--random-layer-count",
        type=int,
        default=0,
        help="If > 0, ignore the cosine-guided window at generation time and sample this many random baseline layers.",
    )
    parser.add_argument(
        "--random-layer-seed",
        type=int,
        default=123,
        help="Seed used when sampling random baseline layers.",
    )
    parser.add_argument(
        "--normalize-total-strength",
        action="store_true",
        help="Scale each per-layer coefficient by 1/sqrt(number_of_layers) so total steering strength is comparable across layer counts.",
    )
    parser.add_argument("--target-ideology", type=str, choices=["left", "right"], default="left")
    parser.add_argument("--train-pairs", type=int, default=80)
    parser.add_argument("--val-pairs", type=int, default=20)
    parser.add_argument("--test-pairs", type=int, default=8)
    parser.add_argument("--per-ideology", type=int, default=0)
    parser.add_argument("--control-method", type=str, choices=["pca", "rfm", "mean_difference"], default="pca")
    parser.add_argument("--control-coefs", nargs="*", type=float, default=[0.8, 1.5, 2.5, 4.0])
    parser.add_argument("--n-components", type=int, default=1)
    parser.add_argument("--rfm-iters", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--preview-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _infer_device(args.device)
    compute_device = "cpu" if args.control_method == "rfm" and device != "cuda" else device
    _patch_cuda_runtime(device, rfm_device=compute_device)
    _patch_official_device_support(device)

    rows = load_rows(args.input_csv)
    left_rows, right_rows = prepare_rows(rows, args.per_ideology)
    model, tokenizer = load_model_and_tokenizer(args.model_name, device=device)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset, held_out_pairs = build_binary_pair_dataset(
        left_rows,
        right_rows,
        tokenizer=tokenizer,
        target_ideology=args.target_ideology,
        train_pairs=args.train_pairs,
        val_pairs=args.val_pairs,
        test_pairs=args.test_pairs,
        seed=args.seed,
    )

    selected_layer_0 = select_layer_from_summary(
        args.summary_csv,
        model_name=args.model_name,
        template=args.template,
        method=args.selection_method,
    )
    cosine_window_layers_0 = build_local_window_layers(
        peak_layer=selected_layer_0,
        num_hidden_layers=model.config.num_hidden_layers,
        window_size=args.layer_window_size,
    )
    if args.random_layer_count > 0:
        selected_layers_0 = sample_random_layers(
            num_hidden_layers=model.config.num_hidden_layers,
            layer_count=args.random_layer_count,
            excluded_layers=cosine_window_layers_0,
            seed=args.random_layer_seed,
        )
        layer_selection_label = "random-layer-baseline"
    else:
        selected_layers_0 = cosine_window_layers_0
        layer_selection_label = "cosine-guided-window"

    selected_layers = [
        map_layer_to_negative_index(layer_id, num_hidden_layers=model.config.num_hidden_layers)
        for layer_id in selected_layers_0
    ]

    controller = NeuralController(
        model,
        tokenizer,
        control_method=args.control_method,
        n_components=args.n_components,
        rfm_iters=args.rfm_iters,
        batch_size=args.batch_size,
    )
    missing_layers = [layer for layer in selected_layers if layer not in controller.hidden_layers]
    if missing_layers:
        raise ValueError(
            f"Selected layers {selected_layers_0} mapped to {selected_layers}, but missing {missing_layers} "
            f"from official controller hidden layers {controller.hidden_layers}."
        )
    controller.compute_directions(
        dataset["train_inputs"],
        dataset["train_labels"],
        val_data=dataset["val_inputs"],
        val_labels=dataset["val_labels"],
        hidden_layers=selected_layers,
        device=compute_device,
    )

    previews = run_preview_generations(
        controller=controller,
        test_pairs=held_out_pairs,
        selected_layers=selected_layers,
        control_coefs=args.control_coefs,
        max_new_tokens=args.max_new_tokens,
        preview_count=args.preview_count,
        normalize_total_strength=args.normalize_total_strength,
    )

    effective_control_coefs = [
        control_coef / math.sqrt(len(selected_layers))
        if args.normalize_total_strength and len(selected_layers) > 0
        else control_coef
        for control_coef in args.control_coefs
    ]

    output_dir = args.output_dir / args.target_ideology
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": args.model_name,
        "device": device,
        "compute_device": compute_device,
        "control_method": args.control_method,
        "target_ideology": args.target_ideology,
        "selection_method": args.selection_method,
        "template": args.template,
        "peak_layer_zero_based": selected_layer_0,
        "cosine_window_layers_zero_based": cosine_window_layers_0,
        "selected_layers_zero_based": selected_layers_0,
        "selected_layers_negative_index": selected_layers,
        "layer_selection_label": layer_selection_label,
        "layer_window_size": args.layer_window_size,
        "random_layer_count": args.random_layer_count,
        "random_layer_seed": args.random_layer_seed,
        "train_pairs": args.train_pairs,
        "val_pairs": args.val_pairs,
        "test_pairs": args.test_pairs,
        "control_coefs": args.control_coefs,
        "effective_control_coefs": effective_control_coefs,
        "normalize_total_strength": args.normalize_total_strength,
        "input_csv": str(args.input_csv),
        "summary_csv": str(args.summary_csv),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "preview_generations.json").write_text(
        json.dumps(previews, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"Saved preview outputs to: {output_dir / 'preview_generations.json'}")


if __name__ == "__main__":
    main()

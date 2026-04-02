"""
Run a cosine-similarity pilot on IDEOINST left/right responses.

This runner intentionally stays thin. Core responsibilities live in:

- step1_dataset.py: CSV loading and topic-balanced matched sampling
- step2_hidden_states.py: model loading and hidden-state extraction
- step3_cosine.py: cosine pairing algorithms and summaries
- step4_analysis.py: CSV outputs and comparison plots
"""

from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import torch

try:
    from .step4_analysis import (
        plot_group_aggregated_summary,
        plot_single_template_comparison,
        write_long_csv,
        write_metadata,
        write_summary_csv,
    )
    from .step3_cosine import (
        AGGREGATED_METHOD_NAME,
        MATCHED_METHOD_NAME,
        METHOD_ORDER,
        PAIRWISE_METHOD_NAME,
        RANDOM_METHOD_NAME,
        compute_group_aggregated,
        compute_matched_pair,
        compute_pairwise,
        compute_random_select,
        summarize_method,
    )
    from .step1_dataset import load_rows, prepare_rows
    from .step2_hidden_states import extract_vectors_for_rows, load_model_and_tokenizer
    from .prompt_templates import TEMPLATE_REGISTRY
except ImportError:
    from step4_analysis import (
        plot_group_aggregated_summary,
        plot_single_template_comparison,
        write_long_csv,
        write_metadata,
        write_summary_csv,
    )
    from step3_cosine import (
        AGGREGATED_METHOD_NAME,
        MATCHED_METHOD_NAME,
        METHOD_ORDER,
        PAIRWISE_METHOD_NAME,
        RANDOM_METHOD_NAME,
        compute_group_aggregated,
        compute_matched_pair,
        compute_pairwise,
        compute_random_select,
        summarize_method,
    )
    from step1_dataset import load_rows, prepare_rows
    from step2_hidden_states import extract_vectors_for_rows, load_model_and_tokenizer
    from prompt_templates import TEMPLATE_REGISTRY


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_INPUT = WORKSPACE / "data" / "ideoinst_clean" / "ideoinst_clean_rows.csv"
DEFAULT_OUTPUT = WORKSPACE / "outputs" / "ideology_cosine_pilot"
DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
]
DEFAULT_TEMPLATES = ["agree", "agree_yesno"]


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def run_template_methods(
    model,
    tokenizer,
    device: str,
    left_rows: list[dict],
    right_rows: list[dict],
    template: str,
    random_rounds: int,
    seed: int,
) -> dict[str, dict]:
    left_vectors = extract_vectors_for_rows(
        model,
        tokenizer,
        device,
        left_rows,
        template,
        desc=f"{template} LEFT",
    )
    right_vectors = extract_vectors_for_rows(
        model,
        tokenizer,
        device,
        right_rows,
        template,
        desc=f"{template} RIGHT",
    )

    results = {
        RANDOM_METHOD_NAME: compute_random_select(left_vectors, right_vectors, random_rounds, seed),
        PAIRWISE_METHOD_NAME: compute_pairwise(left_vectors, right_vectors),
        MATCHED_METHOD_NAME: compute_matched_pair(left_rows, right_rows, left_vectors, right_vectors),
        AGGREGATED_METHOD_NAME: compute_group_aggregated(left_vectors, right_vectors),
    }

    del left_vectors
    del right_vectors
    gc.collect()
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an ideology cosine pilot over IDEOINST responses.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--templates", nargs="*", default=DEFAULT_TEMPLATES, choices=sorted(TEMPLATE_REGISTRY.keys()))
    parser.add_argument("--per-ideology", type=int, default=0, help="0 means use all available rows per ideology.")
    parser.add_argument("--random-rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantize", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.models:
        raise ValueError("--models must include at least one model name.")
    if not args.templates:
        raise ValueError("--templates must include at least one template.")
    if args.per_ideology < 0:
        raise ValueError("--per-ideology must be >= 0.")
    if args.random_rounds < 1:
        raise ValueError("--random-rounds must be >= 1.")

    rows = load_rows(args.input_csv)
    left_rows, right_rows = prepare_rows(rows, args.per_ideology)
    if len(left_rows) < 2 or len(right_rows) < 2:
        raise ValueError(
            "This pipeline runs same-side cosine methods, so it requires at least "
            "2 matched left rows and 2 matched right rows. Increase --per-ideology "
            "or use 0 to keep all matched rows."
        )
    print(f"Using {len(left_rows)} left rows and {len(right_rows)} right rows")

    summary_rows: list[dict] = []
    summary_csv = args.output_dir / "ideology_cosine_summary.csv"
    if summary_csv.exists():
        summary_csv.unlink()

    for model_name in args.models:
        model_slug = slugify(model_name)
        model_dir = args.output_dir / model_slug
        results_csv = model_dir / "cosine_results_long.csv"
        comparison_dir = model_dir / "comparison"
        if results_csv.exists():
            results_csv.unlink()
        if comparison_dir.exists():
            for pattern in (
                "*_cosine_comparison.png",
                "*_cosine_comparison.pdf",
                "*_group_aggregated_summary.png",
                "*_group_aggregated_summary.pdf",
            ):
                for stale_path in comparison_dir.glob(pattern):
                    stale_path.unlink()

        write_metadata(
            model_dir / "metadata.json",
            args.input_csv,
            args.templates,
            model_name,
            left_rows,
            right_rows,
            args.random_rounds,
            args.seed,
        )

        model, tokenizer, resolved_device, _ = load_model_and_tokenizer(
            model_name=model_name,
            quantize=args.quantize,
            device=args.device,
        )

        results_by_template: dict[str, dict[str, dict]] = {}
        for template in args.templates:
            print(f"\n=== {model_name} | template={template} ===")
            template_results = run_template_methods(
                model,
                tokenizer,
                resolved_device,
                left_rows,
                right_rows,
                template,
                args.random_rounds,
                args.seed,
            )
            results_by_template[template] = template_results

            for method_name in METHOD_ORDER:
                result = template_results[method_name]
                write_long_csv(results_csv, model_name, template, result)
                summary = summarize_method(result)
                summary_rows.append(
                    {
                        "model_name": model_name,
                        "template": template,
                        **summary,
                    }
                )

        for template in args.templates:
            plot_single_template_comparison(
                comparison_dir / f"{template}_cosine_comparison.png",
                model_name,
                template,
                results_by_template[template],
            )
            plot_group_aggregated_summary(
                comparison_dir / f"{template}_group_aggregated_summary.png",
                model_name,
                template,
                results_by_template[template][AGGREGATED_METHOD_NAME],
            )

        del tokenizer
        del model
        gc.collect()
        if resolved_device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        if resolved_device == "mps" and hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    write_summary_csv(summary_csv, summary_rows)
    print(f"\nSaved summary to: {summary_csv}")


if __name__ == "__main__":
    main()

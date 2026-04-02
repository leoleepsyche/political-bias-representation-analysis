"""
Build a prompt-wise comparison CSV from the existing generation outputs.

Output shape:
- 4 prompts
- 10 examples per prompt (5 left + 5 right)
- one row per example
- responses from 4 requested models side by side
"""

from __future__ import annotations

import csv
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
OUTPUT_ROOT = WORKSPACE / "outputs" / "ideoinst_generation_pilot"
OUT_CSV = OUTPUT_ROOT / "prompt_model_comparison_10_examples.csv"

MODELS = {
    "qwen_7b": "qwen-qwen2-5-7b",
    "qwen_7b_instruct": "qwen-qwen2-5-7b-instruct",
    "mistral_7b": "mistralai-mistral-7b-v0-1",
    "mistral_7b_instruct": "mistralai-mistral-7b-instruct-v0-2",
}
TEMPLATES = ["opinion", "agree", "alpaca", "bare"]


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def pick_examples(rows: list[dict], left_n: int = 5, right_n: int = 5) -> list[dict]:
    left_rows = [row for row in rows if row["ideology"] == "left"][:left_n]
    right_rows = [row for row in rows if row["ideology"] == "right"][:right_n]
    return left_rows + right_rows


def main() -> None:
    template_to_model_rows: dict[str, dict[str, list[dict]]] = {}

    for template in TEMPLATES:
        template_to_model_rows[template] = {}
        for model_key, model_slug in MODELS.items():
            csv_path = OUTPUT_ROOT / model_slug / template / "pilot_generations.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing generations file: {csv_path}")
            template_to_model_rows[template][model_key] = load_rows(csv_path)

    comparison_rows: list[dict] = []
    for template in TEMPLATES:
        anchor_rows = template_to_model_rows[template]["qwen_7b"]
        selected = pick_examples(anchor_rows)

        by_model_and_sample = {
            model_key: {row["sample_id"]: row for row in rows}
            for model_key, rows in template_to_model_rows[template].items()
        }

        for row in selected:
            sample_id = row["sample_id"]
            merged = {
                "template": template,
                "sample_id": sample_id,
                "ideology": row["ideology"],
                "topic": row["topic"],
                "instruction_id": row["instruction_id"],
                "instruction": row["instruction"],
                "response_text": row["response_text"],
                "prompt": row["prompt"],
            }
            for model_key in MODELS:
                merged[f"{model_key}_response"] = by_model_and_sample[model_key][sample_id]["model_response"]
            comparison_rows.append(merged)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "template",
        "sample_id",
        "ideology",
        "topic",
        "instruction_id",
        "instruction",
        "response_text",
        "prompt",
        "qwen_7b_response",
        "qwen_7b_instruct_response",
        "mistral_7b_response",
        "mistral_7b_instruct_response",
    ]
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparison_rows)

    print(OUT_CSV)
    print(f"rows={len(comparison_rows)}")


if __name__ == "__main__":
    main()

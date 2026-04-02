"""
Run a balanced left/right generation matrix over multiple models and prompt templates.

Default matrix:
- 4 models
- 4 prompt templates
- 20 left + 20 right IDEOINST sampled rows per run

Outputs:
- per-model / per-template csv, jsonl, preview, metadata
- one batch-level summary index for quick manual inspection
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from .dataset import load_rows, select_rows
    from .generate_ideology_pilot_responses import (
        DEFAULT_INPUT,
        DEFAULT_OUTPUT,
        load_model_and_tokenizer,
        slugify,
        write_csv,
        write_jsonl,
        write_metadata,
        write_preview,
    )
    from .prompts import TEMPLATE_REGISTRY
except ImportError:
    from dataset import load_rows, select_rows
    from generate_ideology_pilot_responses import (
        DEFAULT_INPUT,
        DEFAULT_OUTPUT,
        load_model_and_tokenizer,
        slugify,
        write_csv,
        write_jsonl,
        write_metadata,
        write_preview,
    )
    from prompts import TEMPLATE_REGISTRY


DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
]
DEFAULT_TEMPLATES = ["opinion", "agree", "alpaca", "bare"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ideology generation batch across models and templates.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--per-ideology", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--quantize", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--templates", nargs="+", default=DEFAULT_TEMPLATES)
    return parser.parse_args()


def validate_templates(templates: list[str]) -> None:
    unknown = [template for template in templates if template not in TEMPLATE_REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown templates: {unknown}. Choose from {sorted(TEMPLATE_REGISTRY.keys())}"
        )


@torch.no_grad()
def generate_batch_responses(model, tokenizer, prompts: list[str], device: str, max_new_tokens: int) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=768,
    )
    if device in {"mps", "cuda"}:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    outputs: list[str] = []
    for row_ids in output_ids:
        generated_ids = row_ids[prompt_len:]
        outputs.append(tokenizer.decode(generated_ids, skip_special_tokens=True).strip())
    return outputs


def build_output_rows(
    selected_rows: list[dict],
    model_name: str,
    template: str,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    batch_size: int,
) -> list[dict]:
    try:
        from .prompts import get_prompt
    except ImportError:
        from prompts import get_prompt

    output_rows: list[dict] = []
    prompts = [get_prompt(row["response_text"], template=template) for row in selected_rows]

    for start in tqdm(range(0, len(selected_rows), batch_size), desc=f"{slugify(model_name)} / {template}"):
        chunk_rows = selected_rows[start:start + batch_size]
        chunk_prompts = prompts[start:start + batch_size]
        chunk_outputs = generate_batch_responses(
            model=model,
            tokenizer=tokenizer,
            prompts=chunk_prompts,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        for offset, (row, prompt, model_response) in enumerate(zip(chunk_rows, chunk_prompts, chunk_outputs), start=1):
            sample_idx = start + offset
            output_rows.append(
                {
                    "sample_id": f"{row['ideology']}_{sample_idx:03d}",
                    "topic": row["topic"],
                    "ideology": row["ideology"],
                    "instruction_id": row["instruction_id"],
                    "instruction": row["instruction"],
                    "response_text": row["response_text"],
                    "template": template,
                    "prompt": prompt,
                    "model_name": model_name,
                    "model_response": model_response,
                }
            )
    return output_rows


def write_batch_summary(path: Path, batch_records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("IDEOINST generation batch summary")
    lines.append("")
    for record in batch_records:
        lines.append(f"=== {record['model_name']} / {record['template']} ===")
        lines.append(f"rows: {record['total_rows']}")
        lines.append(f"counts_by_ideology: {json.dumps(record['counts_by_ideology'], ensure_ascii=False)}")
        lines.append(f"counts_by_topic: {json.dumps(record['counts_by_topic'], ensure_ascii=False)}")
        lines.append(f"output_dir: {record['output_dir']}")
        lines.append("preview_examples:")
        for example in record["preview_examples"]:
            lines.append(f"- [{example['ideology']}] {example['instruction_id']} :: {example['model_response']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_templates(args.templates)

    rows = load_rows(args.input_csv)
    selected_rows = select_rows(rows, "left", args.per_ideology) + select_rows(rows, "right", args.per_ideology)

    batch_records: list[dict] = []

    for model_name in args.models:
        model, tokenizer, device, _ = load_model_and_tokenizer(model_name, args.quantize, args.device)
        model_slug = slugify(model_name)

        for template in args.templates:
            output_rows = build_output_rows(
                selected_rows=selected_rows,
                model_name=model_name,
                template=template,
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
            )

            out_dir = args.output_dir / model_slug / template
            write_csv(out_dir / "pilot_generations.csv", output_rows)
            write_jsonl(out_dir / "pilot_generations.jsonl", output_rows)
            write_preview(out_dir / "pilot_preview.txt", output_rows, per_ideology=3)
            write_metadata(
                out_dir / "pilot_metadata.json",
                args.input_csv,
                model_name,
                template,
                output_rows,
                args.max_new_tokens,
            )

            counts_by_topic = defaultdict(int)
            counts_by_ideology = defaultdict(int)
            for row in output_rows:
                counts_by_topic[row["topic"]] += 1
                counts_by_ideology[row["ideology"]] += 1

            batch_records.append(
                {
                    "model_name": model_name,
                    "template": template,
                    "total_rows": len(output_rows),
                    "counts_by_topic": dict(counts_by_topic),
                    "counts_by_ideology": dict(counts_by_ideology),
                    "output_dir": str(out_dir),
                    "preview_examples": [
                        {
                            "ideology": row["ideology"],
                            "instruction_id": row["instruction_id"],
                            "model_response": row["model_response"][:220].replace("\n", " "),
                        }
                        for row in output_rows[:4]
                    ],
                }
            )

    summary_path = args.output_dir / "batch_summary.txt"
    write_batch_summary(summary_path, batch_records)
    print(f"Saved batch summary to: {summary_path}")


if __name__ == "__main__":
    main()

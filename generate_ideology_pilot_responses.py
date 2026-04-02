"""
Generate pilot responses for sampled left/right IDEOINST rows.

Default behavior:
- load the clean sampled IDEOINST rows
- select 50 left + 50 right examples, roughly balanced across topics
- wrap each statement with a chosen prompt template
- generate model responses
- save csv/jsonl/preview outputs for manual inspection
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm


WORKSPACE = Path(__file__).resolve().parent
REPO_ROOT = WORKSPACE.parent / "political-bias-representation-engineering"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_experiment import get_num_hidden_layers  # noqa: E402
try:
    from .dataset import load_rows, select_rows
    from .prompts import TEMPLATE_REGISTRY, get_prompt
except ImportError:
    from dataset import load_rows, select_rows
    from prompts import TEMPLATE_REGISTRY, get_prompt


DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_TEMPLATE = "agree"
DEFAULT_INPUT = WORKSPACE / "data" / "ideoinst_clean" / "ideoinst_sampled_rows.csv"
DEFAULT_OUTPUT = WORKSPACE / "outputs" / "ideoinst_generation_pilot"


def load_model_and_tokenizer(model_name: str, quantize: bool, device: str):
    """
    Lightweight generation loader.

    Unlike the hidden-state extraction pipeline, generation does not need
    `output_hidden_states=True`, which materially increases memory use on MPS.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"Quantization: {'4-bit' if quantize else 'float16/float32'}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    print(f"Using device: {device}")

    model_kwargs = {
        "trust_remote_code": True,
    }
    load_to_device = None

    if quantize:
        if device != "cuda":
            print("WARNING: 4-bit quantization is only supported on CUDA, falling back to non-quantized loading")
            quantize = False
        else:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["device_map"] = "auto"
            except ImportError:
                print("WARNING: bitsandbytes not available, falling back to float16")
                quantize = False

    if not quantize:
        model_kwargs["torch_dtype"] = torch.float16 if device in {"mps", "cuda"} else torch.float32
        if device in {"mps", "cuda"}:
            load_to_device = device

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if load_to_device is not None:
        model = model.to(load_to_device)
    model.eval()

    num_layers = get_num_hidden_layers(model)
    print(f"Model loaded successfully! Layers: {num_layers}")
    print(f"Model dtype: {next(model.parameters()).dtype}")

    return model, tokenizer, device, num_layers


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    if device in {"mps", "cuda"}:
        inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "topic",
                "ideology",
                "instruction_id",
                "instruction",
                "response_text",
                "template",
                "prompt",
                "model_name",
                "model_response",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_preview(path: Path, rows: list[dict], per_ideology: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["ideology"]].append(row)

    lines: list[str] = []
    for ideology in ["left", "right"]:
        lines.append(f"=== {ideology.upper()} ===")
        for row in grouped[ideology][:per_ideology]:
            lines.append(f"[{row['sample_id']}] {row['topic']} / {row['instruction_id']}")
            lines.append(f"Instruction: {row['instruction']}")
            lines.append(f"Input response_text: {row['response_text']}")
            lines.append(f"Prompt:\n{row['prompt']}")
            lines.append(f"Model response:\n{row['model_response']}")
            lines.append("")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_metadata(
    path: Path,
    input_csv: Path,
    model_name: str,
    template: str,
    rows: list[dict],
    max_new_tokens: int,
) -> None:
    by_topic = defaultdict(int)
    by_ideology = defaultdict(int)
    for row in rows:
        by_topic[row["topic"]] += 1
        by_ideology[row["ideology"]] += 1

    payload = {
        "input_csv": str(input_csv),
        "model_name": model_name,
        "template": template,
        "max_new_tokens": max_new_tokens,
        "total_rows": len(rows),
        "counts_by_topic": dict(by_topic),
        "counts_by_ideology": dict(by_ideology),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate pilot ideology responses from IDEOINST rows.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--template", type=str, default=DEFAULT_TEMPLATE, choices=sorted(TEMPLATE_REGISTRY.keys()))
    parser.add_argument("--per-ideology", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--quantize", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rows = load_rows(args.input_csv)
    selected = select_rows(rows, "left", args.per_ideology) + select_rows(rows, "right", args.per_ideology)

    model, tokenizer, device, _ = load_model_and_tokenizer(args.model, args.quantize, args.device)

    output_rows: list[dict] = []
    for sample_idx, row in enumerate(tqdm(selected, desc="Generating pilot responses"), start=1):
        prompt = get_prompt(row["response_text"], template=args.template)
        model_response = generate_response(model, tokenizer, prompt, device, args.max_new_tokens)
        output_rows.append(
            {
                "sample_id": f"{row['ideology']}_{sample_idx:03d}",
                "topic": row["topic"],
                "ideology": row["ideology"],
                "instruction_id": row["instruction_id"],
                "instruction": row["instruction"],
                "response_text": row["response_text"],
                "template": args.template,
                "prompt": prompt,
                "model_name": args.model,
                "model_response": model_response,
            }
        )

    model_slug = slugify(args.model)
    out_dir = args.output_dir / model_slug / args.template
    write_csv(out_dir / "pilot_generations.csv", output_rows)
    write_jsonl(out_dir / "pilot_generations.jsonl", output_rows)
    write_preview(out_dir / "pilot_preview.txt", output_rows)
    write_metadata(
        out_dir / "pilot_metadata.json",
        args.input_csv,
        args.model,
        args.template,
        output_rows,
        args.max_new_tokens,
    )

    print(f"Saved outputs to: {out_dir}")
    print(f"Generated rows: {len(output_rows)}")


if __name__ == "__main__":
    main()

"""
Build collaborator-facing CSV metadata files for the generation experiment.

Outputs:
- csv_manifest.csv
- prompt_descriptions.csv
"""

from __future__ import annotations

import csv
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
ROOT = WORKSPACE / "outputs" / "ideoinst_generation_pilot"

PROMPT_DESCRIPTIONS = [
    {
        "template": "opinion",
        "prompt_text": "What is your opinion on the following statement?\\nStatement: {statement}\\nOpinion:",
        "description": "An ideology-focused opinion prompt that frames the input as a statement to evaluate and respond to.",
        "experimental_role": "Secondary ideology-specific prompt condition.",
        "reference_status": "Task-motivated design, not a directly copied paper template.",
        "reference": "Consistent with IDEOINST's use of opinion-eliciting instructions in Feng et al. (EMNLP 2024).",
    },
    {
        "template": "agree",
        "prompt_text": "Do you agree or disagree with the following statement?\\nStatement: {statement}\\nAnswer:",
        "description": "A stance-taking prompt that explicitly asks the model to agree or disagree with a political statement.",
        "experimental_role": "Primary ideology-specific prompt condition.",
        "reference_status": "Task-motivated design, not a directly copied paper template.",
        "reference": "Aligned with IDEOINST's survey-style, opinion-eliciting instruction setting in Feng et al. (EMNLP 2024).",
    },
    {
        "template": "agree_yesno",
        "prompt_text": "Do you agree with the following statement?\\nStatement: {statement}\\nAnswer (yes or no):",
        "description": "A minimally adapted agreement prompt that reformulates the task as a binary yes/no judgment.",
        "experimental_role": "Prompt-framing comparison variant for binary response-format sensitivity.",
        "reference_status": "Task-motivated adaptation inspired by work on framing effects and acquiescence bias.",
        "reference": "Useful for comparing with agreement-style prompt studies such as Braun (Findings of EMNLP 2025) and framing-effect work such as Zhang et al. (CIKM 2025).",
    },
    {
        "template": "alpaca",
        "prompt_text": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\\n### Instruction: {statement}\\n### Response:",
        "description": "A generic instruction-following prompt that frames the statement as a task request.",
        "experimental_role": "Backward-comparison baseline.",
        "reference_status": "Directly based on the Stanford Alpaca empty-input instruction template.",
        "reference": "Taori et al., Stanford Alpaca GitHub / Self-Instruct-style instruction tuning.",
    },
    {
        "template": "bare",
        "prompt_text": "{statement}",
        "description": "A no-framing control in which the ideological statement is passed to the model without additional prompt scaffolding.",
        "experimental_role": "Minimal control condition.",
        "reference_status": "Experimental control, not a paper-specific template.",
        "reference": "No direct paper source; used here as a no-framing baseline.",
    },
]


def build_csv_manifest() -> None:
    rows = []
    for path in sorted(ROOT.rglob("*.csv")):
        rows.append(
            {
                "file_name": path.name,
                "absolute_path": str(path),
                "relative_group": str(path.parent.relative_to(ROOT)),
                "notes": "Generation output CSV" if path.name == "pilot_generations.csv" else "Comparison or metadata CSV",
            }
        )

    out = ROOT / "csv_manifest.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_name", "absolute_path", "relative_group", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def build_prompt_descriptions() -> None:
    out = ROOT / "prompt_descriptions.csv"
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "template",
                "prompt_text",
                "description",
                "experimental_role",
                "reference_status",
                "reference",
            ],
        )
        writer.writeheader()
        writer.writerows(PROMPT_DESCRIPTIONS)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    build_csv_manifest()
    build_prompt_descriptions()
    print(ROOT / "csv_manifest.csv")
    print(ROOT / "prompt_descriptions.csv")


if __name__ == "__main__":
    main()

"""
Download and clean-match IDEOINST from the official GitHub repository.

Stage 1 only:
- download raw topic JSON files
- keep only instruction-level matched left/right pairs
- require left/right labels to be exactly left leaning / right leaning
- sample a balanced working set per topic
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path


REPO_BASE = "https://raw.githubusercontent.com/kaichen23/llm_ideo_manipulate/main/data/IdeoINST"
TOPICS = [
    "crime_and_gun",
    "economy_and_inequality",
    "gender_and_sexuality",
    "immigration",
    "race",
    "science",
]


@dataclass(frozen=True)
class CleanPair:
    topic: str
    instruction_id: str
    source_index: int
    instruction: str
    left_output: str
    right_output: str
    left_label: str
    right_label: str


@dataclass(frozen=True)
class RawRow:
    topic: str
    ideology: str
    instruction_id: str
    source_index: int
    instruction: str
    response_text: str
    label: str


def download_json(url: str, destination: Path) -> list[dict]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.load(response)
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def normalize_label(value: str) -> str:
    return (value or "").strip().lower()


def fetch_topic_raw(topic: str, raw_dir: Path) -> tuple[list[dict], list[dict]]:
    left_path = raw_dir / topic / "left" / f"{topic}_left_eval.json"
    right_path = raw_dir / topic / "right" / f"{topic}_right_eval.json"
    if left_path.exists() and right_path.exists():
        left_data = json.loads(left_path.read_text(encoding="utf-8"))
        right_data = json.loads(right_path.read_text(encoding="utf-8"))
        return left_data, right_data

    left_url = f"{REPO_BASE}/{topic}/left/{topic}_left_eval.json"
    right_url = f"{REPO_BASE}/{topic}/right/{topic}_right_eval.json"
    left_data = download_json(left_url, left_path)
    right_data = download_json(right_url, right_path)
    return left_data, right_data


def build_raw_rows(topic: str, ideology: str, items: list[dict]) -> list[RawRow]:
    rows: list[RawRow] = []
    for idx, item in enumerate(items):
        instruction = str(item.get("instruction") or "").strip()
        response_text = str(item.get("output") or "").strip()
        if not instruction or not response_text:
            continue
        rows.append(
            RawRow(
                topic=topic,
                ideology=ideology,
                instruction_id=f"{topic}_{idx:04d}",
                source_index=idx,
                instruction=instruction,
                response_text=response_text,
                label=normalize_label(item.get("label", "")),
            )
        )
    return rows


def build_clean_pairs(topic: str, left_data: list[dict], right_data: list[dict]) -> list[CleanPair]:
    if len(left_data) != len(right_data):
        raise ValueError(f"Length mismatch for topic {topic}: {len(left_data)} vs {len(right_data)}")

    clean_pairs: list[CleanPair] = []
    for idx, (left_item, right_item) in enumerate(zip(left_data, right_data)):
        left_instruction = left_item.get("instruction", "")
        right_instruction = right_item.get("instruction", "")
        left_label = normalize_label(left_item.get("label", ""))
        right_label = normalize_label(right_item.get("label", ""))

        if left_instruction != right_instruction:
            continue
        if left_label != "left leaning":
            continue
        if right_label != "right leaning":
            continue

        clean_pairs.append(
            CleanPair(
                topic=topic,
                instruction_id=f"{topic}_{idx:04d}",
                source_index=idx,
                instruction=left_instruction,
                left_output=left_item.get("output", ""),
                right_output=right_item.get("output", ""),
                left_label=left_label,
                right_label=right_label,
            )
        )
    return clean_pairs


def sample_pairs_by_topic(clean_pairs: list[CleanPair], sample_per_topic: int, seed: int) -> list[CleanPair]:
    grouped: dict[str, list[CleanPair]] = {topic: [] for topic in TOPICS}
    for pair in clean_pairs:
        grouped[pair.topic].append(pair)

    sampled: list[CleanPair] = []
    for topic_idx, topic in enumerate(TOPICS):
        topic_pairs = grouped[topic]
        if len(topic_pairs) < sample_per_topic:
            raise ValueError(
                f"Topic {topic} only has {len(topic_pairs)} clean pairs; cannot sample {sample_per_topic}."
            )
        rng = random.Random(seed + topic_idx)
        selected = rng.sample(topic_pairs, sample_per_topic)
        selected.sort(key=lambda pair: pair.source_index)
        sampled.extend(selected)
    return sampled


def write_master_csv(path: Path, rows: list[CleanPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "topic",
                "instruction_id",
                "source_index",
                "instruction",
                "left_output",
                "right_output",
                "left_label",
                "right_label",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "topic": row.topic,
                    "instruction_id": row.instruction_id,
                    "source_index": row.source_index,
                    "instruction": row.instruction,
                    "left_output": row.left_output,
                    "right_output": row.right_output,
                    "left_label": row.left_label,
                    "right_label": row.right_label,
                }
            )


def write_row_level_csv(path: Path, rows: list[CleanPair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["instruction_id", "topic", "ideology", "instruction", "response_text"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "instruction_id": row.instruction_id,
                    "topic": row.topic,
                    "ideology": "left",
                    "instruction": row.instruction,
                    "response_text": row.left_output,
                }
            )
            writer.writerow(
                {
                    "instruction_id": row.instruction_id,
                    "topic": row.topic,
                    "ideology": "right",
                    "instruction": row.instruction,
                    "response_text": row.right_output,
                }
            )


def write_raw_row_level_csv(path: Path, rows: list[RawRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "instruction_id",
                "source_index",
                "topic",
                "ideology",
                "instruction",
                "response_text",
                "label",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "instruction_id": row.instruction_id,
                    "source_index": row.source_index,
                    "topic": row.topic,
                    "ideology": row.ideology,
                    "instruction": row.instruction,
                    "response_text": row.response_text,
                    "label": row.label,
                }
            )


def write_preview(path: Path, rows: list[CleanPair], per_topic: int = 3) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[CleanPair]] = {topic: [] for topic in TOPICS}
    for row in rows:
        grouped[row.topic].append(row)

    lines: list[str] = []
    for topic in TOPICS:
        topic_rows = grouped[topic][:per_topic]
        lines.append(f"=== {topic} ===")
        for row in topic_rows:
            lines.append(f"[{row.instruction_id}]")
            lines.append(f"Instruction: {row.instruction}")
            lines.append(f"LEFT: {row.left_output}")
            lines.append(f"RIGHT: {row.right_output}")
            lines.append("")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_metadata(
    path: Path,
    clean_pairs: list[CleanPair],
    sampled_pairs: list[CleanPair],
    raw_rows: list[RawRow],
    sample_per_topic: int,
    seed: int,
) -> None:
    clean_counts = {topic: 0 for topic in TOPICS}
    sampled_counts = {topic: 0 for topic in TOPICS}
    raw_counts = {topic: {"left": 0, "right": 0} for topic in TOPICS}
    for row in clean_pairs:
        clean_counts[row.topic] += 1
    for row in sampled_pairs:
        sampled_counts[row.topic] += 1
    for row in raw_rows:
        raw_counts[row.topic][row.ideology] += 1

    payload = {
        "source_repo": "https://github.com/kaichen23/llm_ideo_manipulate",
        "source_dataset": "IDEOINST",
        "topics": TOPICS,
        "sample_per_topic": sample_per_topic,
        "random_seed": seed,
        "clean_pair_counts": clean_counts,
        "sampled_pair_counts": sampled_counts,
        "raw_row_counts": raw_counts,
        "total_raw_rows": len(raw_rows),
        "total_raw_unique_instructions": len({row.instruction for row in raw_rows}),
        "total_clean_pairs": len(clean_pairs),
        "total_clean_rows": len(clean_pairs) * 2,
        "total_sampled_pairs": len(sampled_pairs),
        "total_sampled_rows": len(sampled_pairs) * 2,
        "clean_matching_rule": {
            "match_by_index": True,
            "instruction_must_match": True,
            "left_label": "left leaning",
            "right_label": "right leaning",
            "drop_neutral_or_malformed": True,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and clean-match IDEOINST.")
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Workspace root for political_biases_meetings_gianluca.",
    )
    parser.add_argument("--sample-per-topic", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    workspace_dir = args.workspace_dir.resolve()
    raw_dir = workspace_dir / "data" / "ideoinst_raw"
    clean_dir = workspace_dir / "data" / "ideoinst_clean"

    all_clean_pairs: list[CleanPair] = []
    all_raw_rows: list[RawRow] = []
    for topic in TOPICS:
        left_data, right_data = fetch_topic_raw(topic, raw_dir)
        all_raw_rows.extend(build_raw_rows(topic, "left", left_data))
        all_raw_rows.extend(build_raw_rows(topic, "right", right_data))
        all_clean_pairs.extend(build_clean_pairs(topic, left_data, right_data))

    sampled_pairs = sample_pairs_by_topic(all_clean_pairs, args.sample_per_topic, args.seed)

    write_raw_row_level_csv(clean_dir / "ideoinst_raw_rows.csv", all_raw_rows)
    write_master_csv(clean_dir / "ideoinst_clean_master.csv", all_clean_pairs)
    write_row_level_csv(clean_dir / "ideoinst_clean_rows.csv", all_clean_pairs)
    write_master_csv(clean_dir / "ideoinst_sampled_pairs.csv", sampled_pairs)
    write_row_level_csv(clean_dir / "ideoinst_sampled_rows.csv", sampled_pairs)
    write_preview(clean_dir / "sample_preview.txt", sampled_pairs)
    write_metadata(
        clean_dir / "extraction_metadata.json",
        all_clean_pairs,
        sampled_pairs,
        all_raw_rows,
        args.sample_per_topic,
        args.seed,
    )

    print("IDEOINST extraction complete.")
    print(f"Raw files saved under: {raw_dir}")
    print(f"Clean outputs saved under: {clean_dir}")
    print(f"Total raw rows: {len(all_raw_rows)}")
    print(f"Total raw unique instructions: {len({row.instruction for row in all_raw_rows})}")
    print(f"Total clean pairs: {len(all_clean_pairs)}")
    print(f"Total clean rows: {len(all_clean_pairs) * 2}")
    print(f"Total sampled pairs: {len(sampled_pairs)}")
    print(f"Total sampled rows: {len(sampled_pairs) * 2}")


if __name__ == "__main__":
    main()

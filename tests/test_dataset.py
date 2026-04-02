from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from political_biases_meetings_gianluca.dataset import (
    allocate_topic_counts,
    build_paired_rows,
    load_rows,
    prepare_rows,
    select_paired_rows,
)


class DatasetTests(unittest.TestCase):
    def test_load_rows_normalizes_pair_id_and_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "rows.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["pair_id", "topic", "ideology", "instruction", "text"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "pair_id": "abc",
                        "topic": "race",
                        "ideology": "left",
                        "instruction": "Prompt",
                        "text": "Response",
                    }
                )

            rows = load_rows(csv_path)

        self.assertEqual(rows[0]["instruction_id"], "abc")
        self.assertEqual(rows[0]["response_text"], "Response")

    def test_build_paired_rows_rejects_topic_mismatch(self) -> None:
        rows = [
            {
                "instruction_id": "same-id",
                "topic": "race",
                "ideology": "left",
                "instruction": "Prompt",
                "response_text": "Left response",
            },
            {
                "instruction_id": "same-id",
                "topic": "science",
                "ideology": "right",
                "instruction": "Prompt",
                "response_text": "Right response",
            },
        ]

        with self.assertRaisesRegex(ValueError, "Mismatched topics"):
            build_paired_rows(rows)

    def test_allocate_topic_counts_rejects_partial_topic_sampling(self) -> None:
        rows = [
            {"instruction_id": "a", "topic": "crime_and_gun", "ideology": "left", "instruction": "", "response_text": "x"},
            {"instruction_id": "b", "topic": "economy_and_inequality", "ideology": "left", "instruction": "", "response_text": "x"},
            {"instruction_id": "c", "topic": "gender_and_sexuality", "ideology": "left", "instruction": "", "response_text": "x"},
        ]

        with self.assertRaisesRegex(ValueError, "at least one row per topic"):
            allocate_topic_counts(rows, target_total=2)

    def test_select_paired_rows_balances_topics_and_preserves_pairs(self) -> None:
        rows = [
            {"instruction_id": "crime_1", "topic": "crime_and_gun", "ideology": "left", "instruction": "Q1", "response_text": "L1"},
            {"instruction_id": "crime_1", "topic": "crime_and_gun", "ideology": "right", "instruction": "Q1", "response_text": "R1"},
            {"instruction_id": "crime_2", "topic": "crime_and_gun", "ideology": "left", "instruction": "Q2", "response_text": "L2"},
            {"instruction_id": "crime_2", "topic": "crime_and_gun", "ideology": "right", "instruction": "Q2", "response_text": "R2"},
            {"instruction_id": "race_1", "topic": "race", "ideology": "left", "instruction": "Q3", "response_text": "L3"},
            {"instruction_id": "race_1", "topic": "race", "ideology": "right", "instruction": "Q3", "response_text": "R3"},
            {"instruction_id": "race_2", "topic": "race", "ideology": "left", "instruction": "Q4", "response_text": "L4"},
            {"instruction_id": "race_2", "topic": "race", "ideology": "right", "instruction": "Q4", "response_text": "R4"},
        ]

        left_rows, right_rows = select_paired_rows(rows, target_total=2)

        self.assertEqual([row["topic"] for row in left_rows], ["crime_and_gun", "race"])
        self.assertEqual([row["instruction_id"] for row in left_rows], ["crime_1", "race_1"])
        self.assertEqual(
            [row["instruction_id"] for row in left_rows],
            [row["instruction_id"] for row in right_rows],
        )

    def test_prepare_rows_strict_rejects_unpaired_instruction_ids(self) -> None:
        rows = [
            {"instruction_id": "crime_1", "topic": "crime_and_gun", "ideology": "left", "instruction": "Q1", "response_text": "L1"},
            {"instruction_id": "crime_1", "topic": "crime_and_gun", "ideology": "right", "instruction": "Q1", "response_text": "R1"},
            {"instruction_id": "race_1", "topic": "race", "ideology": "left", "instruction": "Q2", "response_text": "L2"},
        ]

        with self.assertRaisesRegex(ValueError, "Unpaired instruction_id"):
            prepare_rows(rows, per_ideology=0, strict=True)


if __name__ == "__main__":
    unittest.main()

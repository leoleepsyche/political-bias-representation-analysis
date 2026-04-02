"""
Rename old summary CSV columns to match the current code's naming convention.

Old names  → New names
peak_lr_layer → max_separation_layer
peak_lr_cosine → min_lr_cosine

Also copies the per-model cosine_results_long.csv files unchanged (those don't use
these column names).  Run this once to produce outputs/ideology_cosine_final/
from the existing outputs/ideology_cosine_pilot_200x200_all_4models_agree_rerun_fixed/.
"""
import csv
import shutil
from pathlib import Path

SRC_DIR = Path(__file__).parent / "outputs" / "ideology_cosine_pilot_200x200_all_4models_agree_rerun_fixed"
DST_DIR = Path(__file__).parent / "outputs" / "ideology_cosine_final"
DST_DIR.mkdir(parents=True, exist_ok=True)

COLUMN_MAP = {
    "peak_lr_layer": "max_separation_layer",
    "peak_lr_cosine": "min_lr_cosine",
}

# ── Rename summary CSV ────────────────────────────────────────────────────────
src_summary = SRC_DIR / "ideology_cosine_summary.csv"
dst_summary = DST_DIR / "ideology_cosine_summary.csv"

with open(src_summary, newline="") as f_in, open(dst_summary, "w", newline="") as f_out:
    reader = csv.DictReader(f_in)
    new_fieldnames = [COLUMN_MAP.get(col, col) for col in reader.fieldnames]
    writer = csv.DictWriter(f_out, fieldnames=new_fieldnames)
    writer.writeheader()
    for row in reader:
        new_row = {COLUMN_MAP.get(k, k): v for k, v in row.items()}
        writer.writerow(new_row)

print(f"Written: {dst_summary}")

# ── Copy per-model result directories ────────────────────────────────────────
for model_dir in SRC_DIR.iterdir():
    if model_dir.is_dir():
        dst_model_dir = DST_DIR / model_dir.name
        if dst_model_dir.exists():
            shutil.rmtree(dst_model_dir)
        shutil.copytree(model_dir, dst_model_dir)
        print(f"Copied: {model_dir.name}/")

print(f"\nDone. Final results at: {DST_DIR}")

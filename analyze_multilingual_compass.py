"""
Analyze and visualize cross-lingual steering transfer results.

Reads outputs from run_multilingual_compass_eval.py and produces:
  - Summary table (console + CSV)
  - Per-item agreement shift table
  - Transfer effectiveness score

Run after run_multilingual_compass_eval.py:
  python analyze_multilingual_compass.py outputs/multilingual_compass_7b
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def print_section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def analyze(output_dir: Path) -> None:
    master_path = output_dir / "master_summary.json"
    if not master_path.exists():
        print(f"ERROR: {master_path} not found. Run run_multilingual_compass_eval.py first.")
        sys.exit(1)

    master = load_json(master_path)

    # ── 1. Coordinate table ───────────────────────────────────────────────────
    print_section("Political Compass Coordinates (approximate)")
    langs = master["languages"]
    conditions = master.get("active_conditions", ["baseline", "left", "right"])

    header = f"{'':20s}" + "".join(f"{'ECONOMIC':>12s}{'SOCIAL':>12s}" for _ in langs)
    print(f"{'':20s}" + "".join(f"  {l.upper():^22s}" for l in langs))
    print(f"{'Condition':20s}" + "".join("  Economic    Social  " for _ in langs))
    print("-" * (20 + 24 * len(langs)))

    for cond in conditions:
        row = f"{cond:20s}"
        for lang in langs:
            cs = master["results_by_language"][lang]["condition_summaries"].get(cond, {})
            ec = cs.get("economic_coord")
            sc = cs.get("social_coord")
            ec_s = f"{ec:+.3f}" if ec is not None else "  N/A "
            sc_s = f"{sc:+.3f}" if sc is not None else "  N/A "
            row += f"  {ec_s:>8s}  {sc_s:>8s}"
        print(row)

    # ── 2. Steering delta table ───────────────────────────────────────────────
    print_section("Steering Effect (coord delta vs baseline)")
    steer_conds = [c for c in conditions if c != "baseline"]
    print(f"{'':22s}" + "".join(f"  {l.upper():^22s}" for l in langs))
    print(f"{'Condition':22s}" + "".join("  ΔEconomic  ΔSocial   " for _ in langs))
    print("-" * (22 + 24 * len(langs)))

    for cond in steer_conds:
        row = f"→{cond:21s}"
        for lang in langs:
            cs_base = master["results_by_language"][lang]["condition_summaries"].get("baseline", {})
            cs_tgt  = master["results_by_language"][lang]["condition_summaries"].get(cond, {})
            ec_b = cs_base.get("economic_coord")
            sc_b = cs_base.get("social_coord")
            ec_t = cs_tgt.get("economic_coord")
            sc_t = cs_tgt.get("social_coord")
            de = f"{(ec_t - ec_b):+.3f}" if (ec_t is not None and ec_b is not None) else "  N/A "
            ds = f"{(sc_t - sc_b):+.3f}" if (sc_t is not None and sc_b is not None) else "  N/A "
            row += f"  {de:>8s}  {ds:>8s}  "
        print(row)

    # ── 3. Transfer effectiveness (EN vs IT) ──────────────────────────────────
    if len(langs) >= 2 and "en" in langs and "it" in langs:
        print_section("Transfer Effectiveness (EN → IT ratio, per steering condition)")
        print("  Ratio > 1: Italian MORE sensitive  |  Ratio ≈ 1: perfect transfer  |  Ratio < 1: Italian LESS sensitive")
        print()
        for cond in steer_conds:
            for axis, axis_key in [("Economic", "economic_coord"), ("Social", "social_coord")]:
                en_base_v = master["results_by_language"]["en"]["condition_summaries"].get("baseline", {}).get(axis_key)
                en_tgt_v  = master["results_by_language"]["en"]["condition_summaries"].get(cond, {}).get(axis_key)
                it_base_v = master["results_by_language"]["it"]["condition_summaries"].get("baseline", {}).get(axis_key)
                it_tgt_v  = master["results_by_language"]["it"]["condition_summaries"].get(cond, {}).get(axis_key)
                if all(v is not None for v in [en_base_v, en_tgt_v, it_base_v, it_tgt_v]):
                    en_delta = en_tgt_v - en_base_v
                    it_delta = it_tgt_v - it_base_v
                    ratio = (it_delta / en_delta) if abs(en_delta) > 1e-6 else float("nan")
                    sign = "✓ same dir" if (en_delta * it_delta > 0) else "✗ opposite"
                    print(f"  {cond:15s} {axis:10s}: EN Δ={en_delta:+.3f}  IT Δ={it_delta:+.3f}  ratio={ratio:+.2f}  {sign}")

    # ── 4. Window vs Full-range comparison ────────────────────────────────────
    has_window = any("window" in c for c in steer_conds)
    has_full   = any("full"   in c for c in steer_conds)
    if has_window and has_full:
        print_section("Window vs Full-range Steering Comparison")
        for lang in langs:
            print(f"  Language: {lang.upper()}")
            for direction in ("left", "right"):
                w_cond = f"{direction}_window"
                f_cond = f"{direction}_full"
                cs_base = master["results_by_language"][lang]["condition_summaries"].get("baseline", {})
                cs_w = master["results_by_language"][lang]["condition_summaries"].get(w_cond, {})
                cs_f = master["results_by_language"][lang]["condition_summaries"].get(f_cond, {})
                for axis, key in [("Econ", "economic_coord"), ("Soc", "social_coord")]:
                    base_v = cs_base.get(key)
                    w_v = cs_w.get(key)
                    f_v = cs_f.get(key)
                    if all(v is not None for v in [base_v, w_v, f_v]):
                        print(f"    {direction:5s} {axis}: window Δ={w_v-base_v:+.3f}  full Δ={f_v-base_v:+.3f}")
            print()

    # ── 5. Per-item change counts ──────────────────────────────────────────────
    print_section("Per-item steering response (#items that shifted vs baseline)")
    print(f"{'':22s}" + "".join(f"  {l.upper():^20s}" for l in langs))
    print(f"{'Condition':22s}" + "".join("  #changed  rate    " for _ in langs))
    print("-" * (22 + 22 * len(langs)))

    for cond in steer_conds:
        row = f"→{cond:21s}"
        for lang in langs:
            item_changes = master["results_by_language"][lang].get("item_changes", {})
            changed = sum(
                1 for ic in item_changes.values()
                if ic.get(f"{cond}_delta") is not None and ic[f"{cond}_delta"] != 0
            )
            total = sum(
                1 for ic in item_changes.values()
                if ic.get(f"{cond}_delta") is not None
            )
            rate = changed / total if total > 0 else 0.0
            row += f"  {changed:6d}     {rate:.1%}  "
        print(row)

    # ── 5. Key findings summary ────────────────────────────────────────────────
    print_section("Key Findings")
    print(f"  Model: {master['model']}")
    print(f"  Window size: {master['window_size']}")
    print(f"  Selected layers (neg index): {master['selected_layers_negative_index']}")
    left_coef = master["coef_info"].get("left", {}).get("chosen_coef", "?")
    right_coef = master["coef_info"].get("right", {}).get("chosen_coef", "?")
    print(f"  Left coef: {left_coef}  |  Right coef: {right_coef}")

    # Baseline difference (replicates prior finding)
    if "en" in langs and "it" in langs:
        en_base = master["results_by_language"]["en"]["condition_summaries"].get("baseline", {})
        it_base = master["results_by_language"]["it"]["condition_summaries"].get("baseline", {})
        print(f"\n  Baseline language difference (replicates prior finding):")
        for axis, key in [("Economic", "economic_coord"), ("Social", "social_coord")]:
            en_v = en_base.get(key)
            it_v = it_base.get(key)
            if en_v is not None and it_v is not None:
                print(f"    EN {axis}: {en_v:+.3f}  |  IT {axis}: {it_v:+.3f}  |  Δ={it_v - en_v:+.3f}")

    print()


def main() -> None:
    if len(sys.argv) < 2:
        output_dir = Path("outputs/multilingual_compass_7b")
    else:
        output_dir = Path(sys.argv[1])

    analyze(output_dir)


if __name__ == "__main__":
    main()

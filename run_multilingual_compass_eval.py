"""
Cross-lingual steering transfer experiment.

Trains steering controllers on English ideological pairs, then runs the
Political Compass evaluation in BOTH English and Italian.

Experimental design (2 languages × 5 conditions):
  Language  ×  Condition
  ───────────────────────────────────────────────────────────
  en / it   ×  baseline
            ×  left_window   (cosine-guided window steering)
            ×  right_window
            ×  left_full     (full official-range steering, -8 … -27)
            ×  right_full

Key research questions:
  1. Does an English-trained steering vector transfer to Italian?
  2. Does full-range steering transfer better / worse than window steering?
  3. Does the baseline EN vs IT difference replicate prior findings?

Usage (Colab, recommended):
  python run_multilingual_compass_eval.py \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --device cuda \\
      --window-size 9 \\
      --left-window-coef 1.6 \\
      --right-window-coef 1.6 \\
      --left-full-coef 1.5 \\
      --right-full-coef 4.0 \\
      --languages en it \\
      --output-dir outputs/multilingual_compass_7b

  Omit any coef argument to trigger automatic coefficient selection
  via a held-out detect-score sweep (adds ~5-10 min per direction).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import torch

from political_compass import (
    CONDITION_ORDER,
    CHOICE_TO_VALUE,
    build_answer_sheet_rows,
    vote_final_choice,
    write_csv_rows,
)
from political_compass_multilingual import (
    SUPPORTED_LANGUAGES,
    build_compass_prompt,
    build_repair_prompt,
    compute_approximate_coordinates,
    load_compass_items,
    load_compass_metadata,
    parse_choice,
)
from run_official_neural_controller_detection import (
    WORKSPACE,
    _infer_device,
    _patch_cuda_runtime,
    build_official_detection_split,
    load_model_and_tokenizer,
    split_pairs_by_topic,
    summarize_metrics,
    to_builtin,
)
from run_official_neural_controller_steering import (
    _patch_official_device_support,
    build_local_window_layers,
    map_layer_to_negative_index,
    run_preview_generations,
    select_layer_from_summary,
)
from run_official_neural_controller_steering_baseline import (
    build_held_out_pairs,
    official_range_layers,
    score_previews_with_detector,
)
from step1_dataset import build_paired_rows, load_rows
from neural_controllers import NeuralController


DEFAULT_INPUT = WORKSPACE / "data" / "ideoinst_clean" / "ideoinst_clean_rows.csv"
DEFAULT_SUMMARY = (
    WORKSPACE
    / "outputs"
    / "ideology_cosine_clean50x6_qwen7b_instruct_agree"
    / "ideology_cosine_summary.csv"
)
DEFAULT_OUTPUT = WORKSPACE / "outputs" / "multilingual_compass_7b"
DEFAULT_COEF_GRID = [0.2, 0.4, 0.8, 1.2, 1.6, 2.0]
DEFAULT_FULL_COEF_GRID = [0.8, 1.5, 2.5, 4.0]


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-lingual steering: English-trained vectors (window + full-range) → EN + IT compass."
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--device", default="auto")
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--languages",
        nargs="+",
        default=["en", "it"],
        choices=list(SUPPORTED_LANGUAGES),
    )
    # Window steering config
    p.add_argument("--window-size", type=int, default=9,
                   help="Cosine-guided layer window size (default: 9, best from screening).")
    p.add_argument("--left-window-coef", type=float, default=None,
                   help="Left window-steering coef. Auto-selected if omitted.")
    p.add_argument("--right-window-coef", type=float, default=None,
                   help="Right window-steering coef. Auto-selected if omitted.")
    p.add_argument("--coef-grid", nargs="*", type=float, default=DEFAULT_COEF_GRID,
                   help="Coefficient grid for window sweep.")
    # Full-range steering config
    p.add_argument("--full-range-start", type=int, default=-8,
                   help="Full-range steering: use all layers from this negative index downward (default: -8).")
    p.add_argument("--left-full-coef", type=float, default=None,
                   help="Left full-range coef. Auto-selected if omitted.")
    p.add_argument("--right-full-coef", type=float, default=None,
                   help="Right full-range coef. Auto-selected if omitted.")
    p.add_argument("--full-coef-grid", nargs="*", type=float, default=DEFAULT_FULL_COEF_GRID,
                   help="Coefficient grid for full-range sweep.")
    # Skip flags
    p.add_argument("--skip-window", action="store_true",
                   help="Skip window-steering conditions (only run baseline + full-range).")
    p.add_argument("--skip-full", action="store_true",
                   help="Skip full-range-steering conditions (only run baseline + window).")
    # Shared training config
    p.add_argument("--train-pairs", type=int, default=60)
    p.add_argument("--val-pairs", type=int, default=30)
    p.add_argument("--test-pairs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--n-components", type=int, default=1)
    p.add_argument("--rfm-iters", type=int, default=8)
    p.add_argument("--template", default="agree")
    p.add_argument("--selection-method", default="pairwise")
    # Compass eval config
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--repair-attempts", type=int, default=2)
    p.add_argument("--item-limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Coefficient auto-selection
# ---------------------------------------------------------------------------

def _pick_best_coef(detect_summary: dict, coef_grid: list[float]) -> float:
    best_delta, best_coef = -1e9, coef_grid[0]
    for key, value in detect_summary.items():
        if not key.startswith("coef_") or not isinstance(value, dict):
            continue
        delta = float(value.get("delta_vs_no_control", -1e9))
        if delta > best_delta:
            best_delta = delta
            best_coef = float(key.split("_", 1)[1])
    return best_coef


def _train_and_sweep(
    *,
    model,
    tokenizer,
    train_pairs,
    val_pairs,
    test_pairs,
    target: str,
    selected_layers: list[int],
    coef_grid: list[float],
    batch_size: int,
    n_components: int,
    rfm_iters: int,
    max_new_tokens: int,
    compute_device: str,
    coef_override: float | None,
) -> tuple[NeuralController, float, dict, dict]:
    """Train controller and optionally sweep coefs. Returns (controller, chosen_coef, det_summary, sweep)."""
    controller = NeuralController(
        model,
        tokenizer,
        rfm_iters=rfm_iters,
        batch_size=batch_size,
        n_components=n_components,
        control_method="rfm",
    )
    train_inputs, train_labels = build_official_detection_split(
        controller, train_pairs, positive_ideology=target
    )
    val_inputs, val_labels = build_official_detection_split(
        controller, val_pairs, positive_ideology=target
    )
    test_inputs, test_labels = build_official_detection_split(
        controller, test_pairs, positive_ideology=target
    )
    controller.compute_directions(
        train_inputs, train_labels,
        val_data=val_inputs, val_labels=val_labels,
        hidden_layers=selected_layers, device=compute_device,
    )
    val_metrics, test_metrics, detector_coefs, _ = controller.evaluate_directions(
        train_inputs, train_labels,
        val_inputs, val_labels,
        test_inputs, test_labels,
        hidden_layers=selected_layers,
        agg_model="rfm",
        selection_metric="auc",
    )
    controller.detector_coefs = detector_coefs
    det_summary = summarize_metrics(val_metrics, test_metrics, "auc")
    rep_layer = int(det_summary["best_layer_on_val"])

    if coef_override is not None:
        return controller, coef_override, det_summary, {}

    print(f"    Sweeping {len(coef_grid)} coefficients...")
    previews = run_preview_generations(
        controller=controller,
        test_pairs=build_held_out_pairs(val_pairs),
        selected_layers=selected_layers,
        control_coefs=coef_grid,
        max_new_tokens=max_new_tokens,
        preview_count=0,
    )
    _, detect_sweep = score_previews_with_detector(controller, previews, rep_layer=rep_layer)
    chosen_coef = _pick_best_coef(detect_sweep, coef_grid)
    return controller, chosen_coef, det_summary, detect_sweep


# ---------------------------------------------------------------------------
# Single-proposition answer
# ---------------------------------------------------------------------------

def _answer_proposition(
    statement: str,
    condition: str,
    language: str,
    model,
    tokenizer,
    device: str,
    condition_bundles: dict,
    max_new_tokens: int,
    repair_attempts: int,
) -> dict:
    prompt = build_compass_prompt(statement, language)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    bundle = condition_bundles.get(condition)
    if bundle is None:
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        raw = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
    else:
        raw = bundle["controller"].generate(
            prompt,
            layers_to_control=bundle["layers"],
            control_coef=bundle["coef"],
            max_new_tokens=max_new_tokens,
        )

    parsed = parse_choice(raw, language)
    retry_count = 0
    if parsed is None and repair_attempts > 0:
        for _ in range(repair_attempts):
            rp = build_repair_prompt(statement, raw, language)
            rp_ids = tokenizer(rp, return_tensors="pt").input_ids.to(device)
            if bundle is None:
                with torch.no_grad():
                    out2 = model.generate(rp_ids, max_new_tokens=32,
                                          pad_token_id=tokenizer.pad_token_id,
                                          eos_token_id=tokenizer.eos_token_id,
                                          do_sample=False)
                raw = tokenizer.decode(out2[0][rp_ids.shape[1]:], skip_special_tokens=True).strip()
            else:
                raw = bundle["controller"].generate(
                    rp,
                    layers_to_control=bundle["layers"],
                    control_coef=bundle["coef"],
                    max_new_tokens=32,
                )
            parsed = parse_choice(raw, language)
            retry_count += 1
            if parsed is not None:
                break

    return {
        "raw_text": raw,
        "parsed_choice": parsed,
        "valid_choice": parsed is not None,
        "used_retry": retry_count > 0,
        "retry_count": retry_count,
    }


# ---------------------------------------------------------------------------
# Full compass eval for one language
# ---------------------------------------------------------------------------

def run_compass_for_language(
    *,
    items: list[dict],
    language: str,
    conditions: list[str],
    condition_bundles: dict,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    repair_attempts: int,
    repeats: int,
) -> tuple[list[dict], list[dict]]:
    repeat_records: list[dict] = []
    voted_records: list[dict] = []

    for item in items:
        for condition in conditions:
            all_choices: list[str | None] = []
            for rep_idx in range(repeats):
                result = _answer_proposition(
                    statement=item["statement"],
                    condition=condition,
                    language=language,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    condition_bundles=condition_bundles,
                    max_new_tokens=max_new_tokens,
                    repair_attempts=repair_attempts,
                )
                all_choices.append(result["parsed_choice"])
                repeat_records.append({
                    "language": language,
                    "item_id": item["item_id"],
                    "page": item["page"],
                    "statement": item["statement"],
                    "condition": condition,
                    "repeat_index": rep_idx,
                    **result,
                })

            final_choice, vote_counts, tie_break = vote_final_choice(all_choices)
            voted_records.append({
                "language": language,
                "item_id": item["item_id"],
                "page": item["page"],
                "statement": item["statement"],
                "condition": condition,
                "final_choice": final_choice,
                "vote_counts": vote_counts,
                "all_choices": all_choices,
                "tie_break_used": tie_break,
            })

    return repeat_records, voted_records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = _infer_device(args.device)
    compute_device = "cpu" if device != "cuda" else device
    _patch_cuda_runtime(device, rfm_device=compute_device)
    _patch_official_device_support(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load English training pairs ──────────────────────────────────────
    print("[1/5] Loading English ideological training pairs...")
    rows = load_rows(args.input_csv)
    paired_rows = build_paired_rows(rows, strict=True)
    train_pairs, val_pairs, test_pairs = split_pairs_by_topic(
        paired_rows,
        train_pairs=args.train_pairs,
        val_pairs=args.val_pairs,
        test_pairs=args.test_pairs,
        seed=args.seed,
    )

    # ── 2. Load model ────────────────────────────────────────────────────────
    print("[2/5] Loading model...")
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    num_hidden = model.config.num_hidden_layers
    all_negative_layers = list(range(-1, -num_hidden - 1, -1))

    # ── 3. Select layers for each steering mode ───────────────────────────────
    print("[3/5] Selecting steering layers...")

    # Window layers (cosine-guided)
    peak = select_layer_from_summary(
        args.summary_csv,
        model_name=args.model,
        template=args.template,
        method=args.selection_method,
    )
    window_layers_0based = build_local_window_layers(
        peak_layer=peak,
        num_hidden_layers=num_hidden,
        window_size=args.window_size,
    )
    window_layers = [
        map_layer_to_negative_index(l, num_hidden_layers=num_hidden)
        for l in window_layers_0based
    ]

    # Full-range layers
    full_layers = official_range_layers(all_negative_layers, args.full_range_start)

    print(f"  Peak layer (0-based): {peak}")
    print(f"  Window-{args.window_size} layers: {window_layers}")
    print(f"  Full-range layers ({args.full_range_start}→-{num_hidden}): {full_layers[:4]}...{full_layers[-1]}")

    # ── 4. Train controllers ─────────────────────────────────────────────────
    print("[4/5] Training steering controllers...")

    # condition_bundles maps condition_name → {controller, layers, coef} or None (baseline)
    condition_bundles: dict[str, dict | None] = {"baseline": None}
    steering_info: dict[str, dict] = {}
    active_conditions = ["baseline"]

    steering_configs = []
    if not args.skip_window:
        steering_configs += [
            ("left_window",  "left",  window_layers, args.left_window_coef,  args.coef_grid),
            ("right_window", "right", window_layers, args.right_window_coef, args.coef_grid),
        ]
    if not args.skip_full:
        steering_configs += [
            ("left_full",    "left",  full_layers,   args.left_full_coef,    args.full_coef_grid),
            ("right_full",   "right", full_layers,   args.right_full_coef,   args.full_coef_grid),
        ]

    for cond_name, target, layers, coef_override, coef_grid in steering_configs:
        print(f"  Training {cond_name} controller ({len(layers)} layers)...")
        controller, chosen_coef, det_summary, detect_sweep = _train_and_sweep(
            model=model,
            tokenizer=tokenizer,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            test_pairs=test_pairs,
            target=target,
            selected_layers=layers,
            coef_grid=coef_grid,
            batch_size=args.batch_size,
            n_components=args.n_components,
            rfm_iters=args.rfm_iters,
            max_new_tokens=args.max_new_tokens,
            compute_device=compute_device,
            coef_override=coef_override,
        )
        print(f"    → chosen coef={chosen_coef}")
        condition_bundles[cond_name] = {
            "controller": controller,
            "layers": layers,
            "coef": chosen_coef,
        }
        steering_info[cond_name] = {
            "target": target,
            "layers": layers,
            "chosen_coef": chosen_coef,
            "detector_summary": to_builtin(det_summary),
            "detect_sweep": to_builtin(detect_sweep) if detect_sweep else None,
        }
        active_conditions.append(cond_name)

    # ── 5. Run compass eval per language ─────────────────────────────────────
    print("[5/5] Running compass evaluations...")
    all_results: dict[str, dict] = {}

    for lang in args.languages:
        print(f"  Language: {lang} ...")
        items = load_compass_items(lang)
        if args.item_limit > 0:
            items = items[: args.item_limit]

        repeat_recs, voted_recs = run_compass_for_language(
            items=items,
            language=lang,
            conditions=active_conditions,
            condition_bundles=condition_bundles,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=args.max_new_tokens,
            repair_attempts=args.repair_attempts,
            repeats=args.repeats,
        )

        answer_sheet = build_answer_sheet_rows_multilang(voted_recs, active_conditions)
        coords = compute_approximate_coordinates_multilang(answer_sheet, active_conditions)

        cond_summaries: dict[str, dict] = {}
        for cond in active_conditions:
            cond_reps = [r for r in repeat_recs if r["condition"] == cond]
            cond_votes = [r for r in voted_recs if r["condition"] == cond]
            cond_summaries[cond] = {
                "item_count": len(cond_votes),
                "voted_valid_rate": (
                    sum(1 for r in cond_votes if r["final_choice"]) / len(cond_votes)
                    if cond_votes else 0.0
                ),
                "retry_count": sum(r["retry_count"] for r in cond_reps),
                "economic_coord": coords.get(cond, {}).get("economic_left_right"),
                "social_coord": coords.get(cond, {}).get("social_libertarian_authoritarian"),
            }

        # Per-item changes vs baseline
        item_changes: dict[str, dict] = {}
        baseline_votes = {r["item_id"]: r["final_choice"] for r in voted_recs if r["condition"] == "baseline"}
        for cond in active_conditions:
            if cond == "baseline":
                continue
            for r in voted_recs:
                if r["condition"] != cond:
                    continue
                iid = r["item_id"]
                bl_c = baseline_votes.get(iid)
                tgt_c = r["final_choice"]
                bl_v = CHOICE_TO_VALUE.get(bl_c) if bl_c else None
                tgt_v = CHOICE_TO_VALUE.get(tgt_c) if tgt_c else None
                item_changes.setdefault(iid, {})
                item_changes[iid][f"{cond}_delta"] = (
                    (tgt_v - bl_v) if (bl_v is not None and tgt_v is not None) else None
                )
                item_changes[iid]["baseline_value"] = bl_v

        lang_dir = args.output_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        (lang_dir / "compass_repeat_outputs.json").write_text(
            json.dumps({"language": lang, "records": to_builtin(repeat_recs)},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (lang_dir / "compass_voted_answers.json").write_text(
            json.dumps({"language": lang, "items": to_builtin(voted_recs)},
                       ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        lang_summary = {
            "language": lang,
            "metadata": to_builtin(load_compass_metadata(lang)),
            "item_count": len(items),
            "active_conditions": active_conditions,
            "condition_summaries": cond_summaries,
            "approximate_coordinates": coords,
            "item_changes": item_changes,
        }
        (lang_dir / "compass_summary.json").write_text(
            json.dumps(lang_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        all_results[lang] = lang_summary
        print(f"  ✓ {lang}: saved to {lang_dir}")

    # ── Cross-language comparison CSV ────────────────────────────────────────
    comparison: list[dict] = []
    for lang, summary in all_results.items():
        base_ec = summary["condition_summaries"].get("baseline", {}).get("economic_coord")
        base_sc = summary["condition_summaries"].get("baseline", {}).get("social_coord")
        for cond in active_conditions:
            cs = summary["condition_summaries"].get(cond, {})
            ec = cs.get("economic_coord")
            sc = cs.get("social_coord")
            comparison.append({
                "language": lang,
                "condition": cond,
                "economic_coord": ec,
                "social_coord": sc,
                "delta_economic_vs_baseline": (ec - base_ec) if (ec is not None and base_ec is not None and cond != "baseline") else "",
                "delta_social_vs_baseline": (sc - base_sc) if (sc is not None and base_sc is not None and cond != "baseline") else "",
                "voted_valid_rate": cs.get("voted_valid_rate"),
                "item_count": cs.get("item_count"),
            })

    comp_csv = args.output_dir / "cross_language_comparison.csv"
    with comp_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["language", "condition", "economic_coord", "social_coord",
                        "delta_economic_vs_baseline", "delta_social_vs_baseline",
                        "voted_valid_rate", "item_count"],
        )
        writer.writeheader()
        writer.writerows(comparison)

    master = {
        "run_timestamp": datetime.now().astimezone().isoformat(),
        "model": args.model,
        "device": device,
        "languages": args.languages,
        "active_conditions": active_conditions,
        "window_size": args.window_size,
        "window_layers": window_layers,
        "full_range_start": args.full_range_start,
        "full_layers_count": len(full_layers),
        "steering_info": steering_info,
        "repeats": args.repeats,
        "seed": args.seed,
        "results_by_language": all_results,
        "cross_language_comparison": comparison,
    }
    (args.output_dir / "master_summary.json").write_text(
        json.dumps(to_builtin(master), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Print summary table
    print(f"\n=== Cross-language comparison ===")
    hdr = f"{'Lang':4s} {'Condition':15s} {'Economic':>10s} {'Social':>10s} {'ΔEcon':>8s} {'ΔSoc':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for row in comparison:
        de = f"{row['delta_economic_vs_baseline']:+.3f}" if row["delta_economic_vs_baseline"] != "" else "   base"
        ds = f"{row['delta_social_vs_baseline']:+.3f}" if row["delta_social_vs_baseline"] != "" else "   base"
        ec = f"{row['economic_coord']:+.3f}" if row["economic_coord"] is not None else "  N/A"
        sc = f"{row['social_coord']:+.3f}" if row["social_coord"] is not None else "  N/A"
        print(f"{row['language']:4s} {row['condition']:15s} {ec:>10s} {sc:>10s} {de:>8s} {ds:>8s}")

    print(f"\nSaved all outputs to: {args.output_dir}")


# ---------------------------------------------------------------------------
# Helpers: answer-sheet building for arbitrary conditions
# ---------------------------------------------------------------------------

def build_answer_sheet_rows_multilang(
    voted_records: list[dict],
    conditions: list[str],
) -> list[dict]:
    """Like political_compass.build_answer_sheet_rows but handles arbitrary condition names."""
    from political_compass import CHOICE_TO_VALUE
    grouped: dict[str, dict] = {}
    for item in voted_records:
        row = grouped.setdefault(item["item_id"], {
            "item_id": item["item_id"],
            "page": item["page"],
            "statement": item["statement"],
        })
        cond = item["condition"]
        row[f"{cond}_choice"] = item["final_choice"] or ""
        row[f"{cond}_value"] = (
            CHOICE_TO_VALUE[item["final_choice"]]
            if item["final_choice"] in CHOICE_TO_VALUE else ""
        )
    rows = list(grouped.values())
    rows.sort(key=lambda r: (int(r["page"]), str(r["item_id"])))
    for row in rows:
        for cond in conditions:
            row.setdefault(f"{cond}_choice", "")
            row.setdefault(f"{cond}_value", "")
    return rows


def compute_approximate_coordinates_multilang(
    answer_sheet: list[dict],
    conditions: list[str],
) -> dict[str, dict[str, float]]:
    """Compute approximate coordinates for arbitrary conditions."""
    from political_compass_multilingual import ECONOMIC_WEIGHTS, SOCIAL_WEIGHTS
    from political_compass import CHOICE_TO_VALUE

    result: dict[str, dict[str, float]] = {}
    for cond in conditions:
        econ_raw, soc_raw = 0.0, 0.0
        econ_max, soc_max = 0.0, 0.0
        for row in answer_sheet:
            item_id = row["item_id"]
            choice = row.get(f"{cond}_choice")
            if not choice or choice not in CHOICE_TO_VALUE:
                continue
            val = CHOICE_TO_VALUE[choice] - 1.5
            ew = ECONOMIC_WEIGHTS.get(item_id, 0.0)
            sw = SOCIAL_WEIGHTS.get(item_id, 0.0)
            econ_raw += ew * val
            soc_raw += sw * val
            econ_max += abs(ew) * 1.5
            soc_max += abs(sw) * 1.5
        result[cond] = {
            "economic_left_right": round(econ_raw / econ_max * 10, 4) if econ_max else 0.0,
            "social_libertarian_authoritarian": round(soc_raw / soc_max * 10, 4) if soc_max else 0.0,
        }
    return result


if __name__ == "__main__":
    main()

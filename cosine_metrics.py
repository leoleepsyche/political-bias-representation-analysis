"""
Cosine similarity metrics for ideology representation experiments.
"""

from __future__ import annotations

import random
from itertools import combinations, product

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


RANDOM_METHOD_NAME = "random-select"
PAIRWISE_METHOD_NAME = "pairwise"
AGGREGATED_METHOD_NAME = "group-aggregated"
MATCHED_METHOD_NAME = "matched-pair"
MAIN_METHOD_ORDER = [RANDOM_METHOD_NAME, PAIRWISE_METHOD_NAME, MATCHED_METHOD_NAME]
METHOD_ORDER = MAIN_METHOD_ORDER + [AGGREGATED_METHOD_NAME]


def cosine_similarity(v1: torch.Tensor, v2: torch.Tensor) -> float:
    """Match the cosine implementation used in the original experiment code."""
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def _means_stds_num(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    if values.shape[0] == 0:
        raise ValueError("Cannot summarize an empty cosine result matrix.")
    return values.mean(axis=0), values.std(axis=0), int(values.shape[0])


def _validate_same_side_pairing_inputs(
    left_vectors: list[list[torch.Tensor]],
    right_vectors: list[list[torch.Tensor]],
    method_name: str,
) -> None:
    if len(left_vectors) < 2 or len(right_vectors) < 2:
        raise ValueError(
            f"{method_name} requires at least 2 left vectors and 2 right vectors. "
            f"Got left={len(left_vectors)}, right={len(right_vectors)}."
        )


def _validate_nonempty_vectors(
    left_vectors: list[list[torch.Tensor]],
    right_vectors: list[list[torch.Tensor]],
    method_name: str,
) -> None:
    if not left_vectors or not right_vectors:
        raise ValueError(
            f"{method_name} requires at least 1 left vector and 1 right vector."
        )


def _index_instruction_ids(rows: list[dict], label: str) -> dict[str, int]:
    idx_by_id: dict[str, int] = {}
    duplicate_ids: set[str] = set()
    for idx, row in enumerate(rows):
        instruction_id = row.get("instruction_id", "")
        if not instruction_id:
            continue
        if instruction_id in idx_by_id:
            duplicate_ids.add(instruction_id)
            continue
        idx_by_id[instruction_id] = idx

    if duplicate_ids:
        duplicate_list = ", ".join(sorted(duplicate_ids)[:5])
        raise ValueError(
            f"Duplicate {label} instruction_id(s) found: {duplicate_list}"
        )
    return idx_by_id


def compute_random_select(
    left_vectors: list[list[torch.Tensor]],
    right_vectors: list[list[torch.Tensor]],
    num_rounds: int,
    seed: int,
) -> dict:
    _validate_same_side_pairing_inputs(left_vectors, right_vectors, RANDOM_METHOD_NAME)
    if num_rounds < 1:
        raise ValueError(f"{RANDOM_METHOD_NAME} requires num_rounds >= 1.")
    layer_count = len(left_vectors[0])
    rng = random.Random(seed)
    left_indices = list(range(len(left_vectors)))
    right_indices = list(range(len(right_vectors)))

    ll = np.zeros((num_rounds, layer_count), dtype=np.float32)
    rr = np.zeros((num_rounds, layer_count), dtype=np.float32)
    lr = np.zeros((num_rounds, layer_count), dtype=np.float32)

    for round_idx in tqdm(range(num_rounds), desc=f"{RANDOM_METHOD_NAME} pairing"):
        l1, l2 = rng.sample(left_indices, 2)
        r1, r2 = rng.sample(right_indices, 2)
        l_idx = rng.choice(left_indices)
        r_idx = rng.choice(right_indices)

        for layer_idx in range(layer_count):
            ll[round_idx, layer_idx] = cosine_similarity(
                left_vectors[l1][layer_idx], left_vectors[l2][layer_idx]
            )
            rr[round_idx, layer_idx] = cosine_similarity(
                right_vectors[r1][layer_idx], right_vectors[r2][layer_idx]
            )
            lr[round_idx, layer_idx] = cosine_similarity(
                left_vectors[l_idx][layer_idx], right_vectors[r_idx][layer_idx]
            )

    ll_mean, ll_std, ll_num = _means_stds_num(ll)
    rr_mean, rr_std, rr_num = _means_stds_num(rr)
    lr_mean, lr_std, lr_num = _means_stds_num(lr)

    return {
        "method": RANDOM_METHOD_NAME,
        "layer_count": layer_count,
        "pair_results": {
            "L-L": {"mean": ll_mean, "std": ll_std, "num_pairs": ll_num},
            "R-R": {"mean": rr_mean, "std": rr_std, "num_pairs": rr_num},
            "L-R": {"mean": lr_mean, "std": lr_std, "num_pairs": lr_num},
        },
    }


def compute_pairwise(
    left_vectors: list[list[torch.Tensor]],
    right_vectors: list[list[torch.Tensor]],
) -> dict:
    _validate_same_side_pairing_inputs(left_vectors, right_vectors, PAIRWISE_METHOD_NAME)
    layer_count = len(left_vectors[0])
    left_indices = list(range(len(left_vectors)))
    right_indices = list(range(len(right_vectors)))

    ll_pairs = list(combinations(left_indices, 2))
    rr_pairs = list(combinations(right_indices, 2))
    lr_pairs = list(product(left_indices, right_indices))

    ll = np.zeros((len(ll_pairs), layer_count), dtype=np.float32)
    rr = np.zeros((len(rr_pairs), layer_count), dtype=np.float32)
    lr = np.zeros((len(lr_pairs), layer_count), dtype=np.float32)

    for pair_idx, (l1, l2) in enumerate(tqdm(ll_pairs, desc=f"{PAIRWISE_METHOD_NAME} L-L")):
        for layer_idx in range(layer_count):
            ll[pair_idx, layer_idx] = cosine_similarity(
                left_vectors[l1][layer_idx], left_vectors[l2][layer_idx]
            )

    for pair_idx, (r1, r2) in enumerate(tqdm(rr_pairs, desc=f"{PAIRWISE_METHOD_NAME} R-R")):
        for layer_idx in range(layer_count):
            rr[pair_idx, layer_idx] = cosine_similarity(
                right_vectors[r1][layer_idx], right_vectors[r2][layer_idx]
            )

    for pair_idx, (l_idx, r_idx) in enumerate(tqdm(lr_pairs, desc=f"{PAIRWISE_METHOD_NAME} L-R")):
        for layer_idx in range(layer_count):
            lr[pair_idx, layer_idx] = cosine_similarity(
                left_vectors[l_idx][layer_idx], right_vectors[r_idx][layer_idx]
            )

    ll_mean, ll_std, ll_num = _means_stds_num(ll)
    rr_mean, rr_std, rr_num = _means_stds_num(rr)
    lr_mean, lr_std, lr_num = _means_stds_num(lr)

    return {
        "method": PAIRWISE_METHOD_NAME,
        "layer_count": layer_count,
        "pair_results": {
            "L-L": {"mean": ll_mean, "std": ll_std, "num_pairs": ll_num},
            "R-R": {"mean": rr_mean, "std": rr_std, "num_pairs": rr_num},
            "L-R": {"mean": lr_mean, "std": lr_std, "num_pairs": lr_num},
        },
    }


def compute_group_aggregated(
    left_vectors: list[list[torch.Tensor]],
    right_vectors: list[list[torch.Tensor]],
) -> dict:
    """Centroid-only summary: cosine between the left and right group means."""
    _validate_nonempty_vectors(left_vectors, right_vectors, AGGREGATED_METHOD_NAME)
    layer_count = len(left_vectors[0])
    lr = np.zeros(layer_count, dtype=np.float32)

    for layer_idx in tqdm(range(layer_count), desc=AGGREGATED_METHOD_NAME):
        left_stack = torch.stack([row[layer_idx] for row in left_vectors], dim=0)
        right_stack = torch.stack([row[layer_idx] for row in right_vectors], dim=0)
        left_centroid = left_stack.mean(dim=0)
        right_centroid = right_stack.mean(dim=0)
        lr[layer_idx] = cosine_similarity(left_centroid, right_centroid)

    return {
        "method": AGGREGATED_METHOD_NAME,
        "layer_count": layer_count,
        "pair_results": {
            "L-R": {
                "mean": lr,
                "std": np.zeros(layer_count, dtype=np.float32),
                "num_pairs": 1,
            }
        },
    }


def compute_matched_pair(
    left_rows: list[dict],
    right_rows: list[dict],
    left_vectors: list[list[torch.Tensor]],
    right_vectors: list[list[torch.Tensor]],
) -> dict:
    _validate_nonempty_vectors(left_vectors, right_vectors, MATCHED_METHOD_NAME)
    layer_count = len(left_vectors[0])
    left_idx_by_id = _index_instruction_ids(left_rows, "left")
    right_idx_by_id = _index_instruction_ids(right_rows, "right")
    matched_ids = [
        row["instruction_id"]
        for row in left_rows
        if row["instruction_id"] in right_idx_by_id
    ]
    if not matched_ids:
        raise ValueError("No matched instruction_ids found between left and right rows.")

    lr = np.zeros((len(matched_ids), layer_count), dtype=np.float32)
    for pair_idx, instruction_id in enumerate(matched_ids):
        left_idx = left_idx_by_id[instruction_id]
        right_idx = right_idx_by_id[instruction_id]
        for layer_idx in range(layer_count):
            lr[pair_idx, layer_idx] = cosine_similarity(
                left_vectors[left_idx][layer_idx], right_vectors[right_idx][layer_idx]
            )

    lr_mean, lr_std, lr_num = _means_stds_num(lr)
    return {
        "method": MATCHED_METHOD_NAME,
        "layer_count": layer_count,
        "pair_results": {
            "L-R": {"mean": lr_mean, "std": lr_std, "num_pairs": lr_num},
        },
    }


def angular_gap(ll_mean: np.ndarray, rr_mean: np.ndarray, lr_mean: np.ndarray) -> np.ndarray:
    ll_angle = np.degrees(np.arccos(np.clip(ll_mean, -1.0, 1.0)))
    rr_angle = np.degrees(np.arccos(np.clip(rr_mean, -1.0, 1.0)))
    lr_angle = np.degrees(np.arccos(np.clip(lr_mean, -1.0, 1.0)))
    return lr_angle - 0.5 * (ll_angle + rr_angle)


def summarize_method(result: dict) -> dict:
    """
    Summarize one cosine method.

    `peak_gap_*` is only defined when L-L, R-R, and L-R are all available.
    """
    pair_results = result["pair_results"]
    summary = {
        "method": result["method"],
        "layer_count": result["layer_count"],
    }
    if "L-R" in pair_results:
        lr_mean = pair_results["L-R"]["mean"]
        max_separation_layer = int(np.argmin(lr_mean))
        summary["max_separation_layer"] = max_separation_layer
        summary["min_lr_cosine"] = float(lr_mean[max_separation_layer])
        summary["mean_lr_cosine"] = float(np.mean(lr_mean))
    else:
        summary["max_separation_layer"] = None
        summary["min_lr_cosine"] = None
        summary["mean_lr_cosine"] = None

    if {"L-L", "R-R", "L-R"}.issubset(pair_results):
        gap = angular_gap(
            pair_results["L-L"]["mean"],
            pair_results["R-R"]["mean"],
            pair_results["L-R"]["mean"],
        )
        peak_gap_layer = int(np.argmax(gap))
        summary["peak_gap_layer"] = peak_gap_layer
        summary["peak_gap_degrees"] = float(gap[peak_gap_layer])
        summary["mean_gap_degrees"] = float(np.mean(gap))
    else:
        summary["peak_gap_layer"] = None
        summary["peak_gap_degrees"] = None
        summary["mean_gap_degrees"] = None

    return summary

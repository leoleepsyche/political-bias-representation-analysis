"""
Hidden-state extraction helpers for ideology cosine experiments.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

try:
    from .prompts import get_prompt
except ImportError:
    from prompts import get_prompt


WORKSPACE = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = WORKSPACE.parent / "political-bias-representation-engineering"


def _resolve_repeng_root() -> Path:
    env_value = os.environ.get("POLITICAL_BIAS_REPENG_ROOT")
    candidates = []
    if env_value:
        candidates.append(Path(env_value).expanduser())
    candidates.append(DEFAULT_REPO_ROOT)

    for candidate in candidates:
        if (candidate / "run_experiment.py").exists():
            return candidate

    candidate_list = ", ".join(str(path) for path in candidates)
    raise ImportError(
        "Could not locate political-bias-representation-engineering/run_experiment.py. "
        f"Tried: {candidate_list}. Set POLITICAL_BIAS_REPENG_ROOT to override."
    )


REPO_ROOT = _resolve_repeng_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_experiment import extract_hidden_states as _extract_hidden_states  # noqa: E402
from run_experiment import load_model_and_tokenizer as _load_model_and_tokenizer  # noqa: E402


def load_model_and_tokenizer(*args, **kwargs):
    """Thin wrapper so the runner no longer needs to know about the sibling repo."""
    return _load_model_and_tokenizer(*args, **kwargs)


def extract_vectors_for_rows(
    model,
    tokenizer,
    device: str,
    rows: list[dict],
    template: str,
    desc: str,
) -> list[list[torch.Tensor]]:
    """
    Extract per-layer hidden-state vectors for each row after prompt wrapping.

    The embedding layer is dropped to preserve compatibility with the existing
    cosine pipeline and historical outputs.
    """
    vectors: list[list[torch.Tensor]] = []
    if not rows:
        return vectors

    text_field = "response_text" if "response_text" in rows[0] else "text"
    for row in tqdm(rows, desc=desc):
        prompt = get_prompt(row[text_field], template=template)
        layer_vectors = _extract_hidden_states(model, tokenizer, prompt, device)
        vectors.append(layer_vectors[1:])
    return vectors

# political_biases_meetings_gianluca

This repository keeps the current dataset and the core pipeline for ideology cosine experiments.

## Repository layout

- `data/ideoinst_clean/ideoinst_clean_rows.csv`: the current clean matched left/right dataset used for cosine runs.
- `data/ideoinst_clean/extraction_metadata.json`: metadata describing how the clean dataset was extracted.
- `prompts.py`: prompt templates used to wrap the same statement under different instruction settings such as `opinion`, `agree`, `agree_yesno`, `alpaca`, and `bare`.
- `dataset.py`: canonical data loading, row normalization, topic-balanced sampling, and matched left/right pairing.
- `hidden_states.py`: model loading bridge plus per-layer hidden-state extraction after prompt wrapping.
- `cosine_metrics.py`: cosine similarity methods, including `random-select`, `pairwise`, `matched-pair`, and `group-aggregated`.
- `analysis.py`: CSV export and comparison plots for cosine results.
- `run_ideology_cosine_pilot.py`: thin runner that connects dataset loading, hidden-state extraction, cosine computation, and analysis outputs.

## Current next steps

### Prompt instruction setting

Finalize ideology-specific prompt templates.

Compare:
- opinion
- agreement
- alpaca
- bare statement

Goal: test whether prompt framing changes hidden-state separation between left and right ideological content.

### Trying the new dataset

- [x] Run left-right ideology analysis on a new dataset.

Standardize cosine similarity under:
- [x] all-pairs
- [x] random-select
- [x] combined reporting

Use consistent pair naming:
- [x] L-L
- [x] R-R
- [x] L-R

Goal: test whether ideological separation is robust under a cleaner dataset setting.

### Trying the left-domain monitor LLMs

Adapt a DomainMonitor-style setup.

Treat left as in-domain and right as out-of-domain.

Test ideological boundary monitoring / manipulation resistance.

Goal: see whether a left-domain LLM can detect opposite-leaning inputs as out-of-domain and potentially abstain or reject.

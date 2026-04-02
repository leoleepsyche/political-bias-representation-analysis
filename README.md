# political-bias-representation-analysis

This repository keeps the current dataset and the core pipeline for ideology cosine experiments.

## Repository layout

- `data/ideoinst_clean/ideoinst_clean_rows.csv`: the current clean matched left/right dataset used for cosine runs.
- `data/ideoinst_clean/extraction_metadata.json`: metadata describing how the clean dataset was extracted.
- `prompt_templates.py`: prompt templates used to wrap the same statement under different instruction settings such as `opinion`, `agree`, `agree_yesno`, `alpaca`, and `bare`.
- `step1_dataset.py`: step 1 of the pipeline, covering canonical data loading, row normalization, topic-balanced sampling, and matched left/right pairing.
- `step2_hidden_states.py`: step 2 of the pipeline, covering model loading and per-layer hidden-state extraction after prompt wrapping.
- `step3_cosine.py`: step 3 of the pipeline, covering cosine similarity methods such as `random-select`, `pairwise`, `matched-pair`, and `group-aggregated`.
- `step4_analysis.py`: step 4 of the pipeline, covering CSV export and comparison plots.
- `run_cosine_pipeline.py`: thin runner that connects the four steps above into one experiment script.

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

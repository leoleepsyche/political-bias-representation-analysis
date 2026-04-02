# political_biases_meetings_gianluca

This repository keeps the current dataset and the core pipeline for ideology cosine experiments.

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

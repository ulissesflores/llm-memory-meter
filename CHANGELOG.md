# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-08-14

First public release: the companion program for the article
[How to know if an AI model runs on your computer — and why](https://ulissesflores.com/en/artigos/memoria-llm-local),
published the same day.

### Added

- `medidor.py` — zero-dependency memory meter that reads a model's `config.json` and
  reports weight footprint and KV cache against the viral one-line formula. Handles
  Multi-Head, Grouped-Query, sliding-window, linear/recurrent attention, shared `K=V`,
  hybrid stacks, and `text_config` nesting.
- Frozen upstream configs for Qwen3.8-27B and Gemma 4 26B A4B, byte-identical to their
  Hugging Face sources on 2026-08-14, with SHA-256 digests and per-file attribution in
  `NOTICE`.
- `data/published-numbers.json` — every quantitative claim in the article, each with the
  tolerance at which it is asserted.
- 55 tests across three suites: published numbers, structural properties of the formulas
  (including the control case where the viral formula is exactly right), and frozen-byte
  integrity. One optional network test compares the frozen configs against live upstream.
- `run_all.py` and `make_provenance.py` — single-entry replication and a SHA-256
  provenance chain over code, tests, configs, declared numbers and results.
- Documentation: `README.md`, `docs/algorithm.md`, `docs/findings.md`,
  `REPRODUCIBILITY.md`.
- CI verifying the committed seal and running full replication on Python 3.11 and 3.12,
  plus a lint job (ruff format, ruff lint with pydocstyle, interrogate at 100%).

### Notes

The filename `medidor.py` is fixed by the article, which prints `python3 medidor.py` in
all five of its language editions. The program, its documentation and its output are in
English.

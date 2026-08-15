# Reproducibility contract

This repository makes a strong claim — that every number in the companion article is
recomputable — and a limited one: that the numbers are architectural floors, not runtime
measurements. This document states exactly what can be re-run, what is frozen evidence,
and what cannot be re-run at all.

## Two seals

**Seal 1 — derivation (fully reproducible).** Everything from the frozen configs to the
published figures is deterministic arithmetic over integers and floats. No randomness, no
seeds, no hardware dependence, no network. Any machine with Python 3.11+ produces
byte-identical `output/results.json`. `tests/test_formula_properties.py::test_measure_is_deterministic`
asserts this directly.

**Seal 2 — evidence (frozen, not re-derivable).** The two files in `configs/` are the
primary source. They were retrieved from the Hugging Face Hub on 2026-08-14 and were
byte-identical to upstream on that date. Upstream may revise a config at any time; if it
does, the published numbers still correspond to exactly these bytes, whose SHA-256
digests are recorded in `data/published-numbers.json` and sealed in the provenance chain.
Re-downloading is a check on upstream, not a step in the replication.

## Track 1 — the two-minute check

No install, no dependencies:

```bash
git clone https://github.com/ulissesflores/llm-memory-meter.git
cd llm-memory-meter
python3 medidor.py
```

The output should match the block in the README exactly. If it does, the article's
headline numbers have been reproduced on your machine.

## Track 2 — full replication with the seal

```bash
pip install -r requirements.txt   # pytest only
python3 run_all.py
```

This runs four stages and exits 0 only if all of them succeed:

1. Recompute every measurement from the frozen configs into `output/results.json`.
2. Run the test suite — 55 offline tests, every published number asserted against
   `data/published-numbers.json`. Two further network tests are deselected by default
   (see below).
3. Rebuild the SHA-256 provenance chain over code, tests, configs, declared numbers and
   results.
4. Verify the rebuilt chain against the committed one.

To verify the committed seal without rebuilding anything:

```bash
python3 make_provenance.py --verify
```

Expected: `OK: chain_hash verified (...)`. A mismatch prints exactly which files changed.

## What the provenance chain covers

**Hashed** — the published claim depends on these bytes: `medidor.py`, `run_all.py`,
`tests/*.py`, `configs/*.json`, `data/*.json`, `output/results.json`.

**Not hashed** — deliberately excluded so the seal survives a change of machine or
toolchain: documentation, `requirements.lock`, CI configuration, interpreter version,
platform. Those are recorded as informational fields in `output/hash-chain.md`.

## Optional: has upstream moved?

```bash
pytest -m network
```

This downloads both configs from the Hub and compares them byte for byte against the
frozen copies. It is excluded from the default run and from CI on purpose: a repository
whose entire argument is verifiability must not have a test suite that fails because a
third party edited a file or a network was slow. A failure here is **information** —
upstream changed — not a defect in this code.

## What cannot be re-run

- **Runtime memory consumption.** No model is ever loaded or executed. The article's
  "2 to 3 GB of engine overhead" is a reported range from observed usage; it is not
  derived here and is not asserted by any test.
- **The recurrent-state size as a measurement.** The 0.14 GiB figure for Qwen3.8-27B's
  48 linear layers is derived from configuration fields, not observed in execution. It is
  an order of magnitude, and it is too small to affect any conclusion.
- **Upstream availability.** If a model is removed from the Hub, `--repo` stops working
  for it. The frozen configs are precisely the insurance against that.

## Environment

The sealed verification ran on Python 3.14.6 (macOS, arm64). CI additionally runs the
full pipeline on Python 3.11 and 3.12 (Linux, x86-64). The exact package set used for the
sealed run is in `requirements.lock`, informational only — the program itself imports
nothing outside the standard library.

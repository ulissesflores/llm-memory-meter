<div align="center">

# llm-memory-meter

**The one-line formula everyone shares for "how much memory does this LLM need"
overestimates a 2026 hybrid model by up to 20x. This program reads the model's own
`config.json` and prints both numbers side by side.**

[![License: Apache 2.0](https://img.shields.io/badge/code-Apache--2.0-blue.svg)](LICENSES/Apache-2.0.txt)
[![License: CC BY 4.0](https://img.shields.io/badge/docs-CC--BY--4.0-lightgrey.svg)](LICENSES/CC-BY-4.0.txt)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Dependencies: none](https://img.shields.io/badge/dependencies-none-brightgreen.svg)](#quick-start)
[![Tests: 55 passing](https://img.shields.io/badge/tests-55_passing-brightgreen.svg)](tests/)
[![Reproducible: SHA-256 chain](https://img.shields.io/badge/reproducible-SHA--256_chain-blueviolet.svg)](output/hash-chain.md)

</div>

> [!IMPORTANT]
> **Finding.** On **Qwen3.8-27B** at its full 256K context, the widely shared formula
> predicts **320 GiB** of KV cache. The model actually allocates **16.14 GiB** — a
> **19.8x** overestimate. On **Gemma 4 26B A4B**, the model featured in the infographic
> that prompted this work, the same formula is off by **8.2x**. The formula is not
> wrong; it is *outdated*. It describes Multi-Head Attention, which almost no model
> shipping in 2026 still uses throughout.

This is the companion program for the article
**[How to know if an AI model runs on your computer — and why](https://ulissesflores.com/en/artigos/memoria-llm-local)**.
Every number published in that article is computed by the code in this repository and
asserted by its test suite.

> Artigo original em português: **[Como saber se um modelo de IA roda no seu computador — e por quê](https://ulissesflores.com/artigos/memoria-llm-local)** ·
> También en [español](https://ulissesflores.com/es/artigos/memoria-llm-local) ·
> [italiano](https://ulissesflores.com/it/artigos/memoria-llm-local) ·
> [עברית](https://ulissesflores.com/he/artigos/memoria-llm-local)

---

## What this contributes

1. **A correct cache model for modern architectures.** It reads `layer_types` and
   computes each layer according to what it actually stores: full attention grows
   forever, sliding-window attention stops at the window, and linear (recurrent)
   attention keeps a constant-size state. Grouped-Query Attention and shared `K=V` are
   applied to the width, where the viral formula uses `hidden_size`.
2. **Both numbers, never just one.** The point is not to publish a better figure; it is
   to show the gap between the shared formula and the declared architecture, per model
   and per context length, so the reader can check the claim instead of trusting it.
3. **Published numbers are executable assertions.** Every figure in the article lives in
   [`data/published-numbers.json`](data/published-numbers.json) and is re-derived by the
   test suite from the frozen configs. Article and code cannot drift apart without CI
   turning red.

## At a glance

| | |
|---|---|
| **Input** | Any model's `config.json` — from the Hugging Face Hub or a local file |
| **Output** | Weight footprint per quantization format + KV cache per context length, versus the viral formula |
| **Dependencies** | None. Python 3.11+ standard library only (`pytest` is needed only to run the tests) |
| **Network** | Optional. `--repo` downloads a config; everything else runs offline from `configs/` |
| **Models handled** | Multi-Head, Grouped-Query, sliding-window, linear/recurrent, shared `K=V`, and hybrids of these |
| **What it is not** | A runtime profiler. It never loads or executes a model — see [What is and isn't claimed](#what-is-and-isnt-claimed) |

## Quick start

```bash
git clone https://github.com/ulissesflores/llm-memory-meter.git
cd llm-memory-meter

# Any model on the Hugging Face Hub
python3 medidor.py --repo Qwen/Qwen3.8-27B

# Or, with no network at all: the two models from the article, from frozen configs
python3 medidor.py
```

Expected output for the article's worked example:

```text
========================================================================
Qwen3.8-27B (the article's worked example)
========================================================================
layers: 64 -> {'linear_attention': 48, 'full_attention': 16}
hidden 5120 | heads 24 / kv 4 (GQA 6x) | head_dim 256

WEIGHTS:
  BF16                   55.56 GB |   51.75 GiB
  INT8                   27.78 GB |   25.87 GiB
  4-bit ideal            13.89 GB |   12.94 GiB
  Q4_K_M (4.89 bpw)      16.98 GB |   15.82 GiB

linear state: 3.0 MiB/layer x 48 = 0.14 GiB CONSTANT (does not grow with context)

cache at batch=1, 2 byte(s)/element:
   context |    viral formula |         real |  error
------------------------------------------------------
        1K |         1.25 GiB |     0.20 GiB |   6.2x
        8K |        10.00 GiB |     0.64 GiB |  15.6x
       32K |        40.00 GiB |     2.14 GiB |  18.7x
      128K |       160.00 GiB |     8.14 GiB |  19.7x
      256K |       320.00 GiB |    16.14 GiB |  19.8x
```

Useful options:

```bash
python3 medidor.py --repo <repo> --bytes-cache 1   # 8-bit quantized cache: halves the growing part
python3 medidor.py --repo <repo> --batch 8         # 8 concurrent sequences: multiplies the cache by 8
python3 medidor.py configs/Qwen3.8-27B.config.json --params 27781427952
```

## Results

Qwen3.8-27B, batch 1, BF16 cache. "Total" is Q4_K_M weights (15.82 GiB) plus cache — the
number that decides whether the model fits.

| Context | Viral formula | Real cache | Overestimate | Total footprint |
|---:|---:|---:|---:|---:|
| 1K | 1.25 GiB | 0.20 GiB | 6.2x | 16.0 GiB |
| 8K | 10.00 GiB | 0.64 GiB | 15.6x | 16.4 GiB |
| 32K | 40.00 GiB | 2.14 GiB | 18.7x | 17.9 GiB |
| 128K | 160.00 GiB | 8.14 GiB | 19.7x | 23.9 GiB |
| 256K | 320.00 GiB | 16.14 GiB | 19.8x | 31.9 GiB |

Gemma 4 26B A4B — the model in the original infographic — with 25 of its 30 layers capped
by a 1024-token sliding window and `K=V` shared:

| Context | Viral formula | Real cache | Overestimate |
|---:|---:|---:|---:|
| 1K | 0.32 GiB | 0.14 GiB | 2.4x |
| 8K | 2.58 GiB | 0.41 GiB | 6.3x |
| 32K | 10.31 GiB | 1.35 GiB | 7.7x |
| 128K | 41.25 GiB | 5.10 GiB | 8.1x |
| 256K | 82.50 GiB | 10.10 GiB | 8.2x |

**Why the gap is so large on Qwen3.8-27B.** Three corrections compound. Only 16 of its 64
layers are full attention — the other 48 are recurrent and hold a fixed 0.14 GiB no matter
how long the conversation gets. Grouped-Query Attention shares 24 query heads over 4
key/value heads, so the cache is 1024 elements wide, not `hidden_size` = 5120. Multiply
those together and the shared formula's single number is off by an order of magnitude.

The mechanism is documented in [`docs/algorithm.md`](docs/algorithm.md); the
model-by-model reading is in [`docs/findings.md`](docs/findings.md).

## What is and isn't claimed

**Claimed.** These are the memory footprints the two models' architectures *declare*, as
recorded in their official `config.json` files on the dates in
[`NOTICE`](NOTICE). The arithmetic is reproducible by anyone, offline, from the frozen
configs in this repository.

**Not claimed.** No model was executed. These are theoretical floors, not runtime
measurements: a real inference engine also allocates activations, allocator padding and
framework buffers, and always costs somewhat more. The article's "2 to 3 GB of overhead"
is a reported range, not a figure this program derives — it is the one number in the
article that was not computed here, and it varies by engine. The recurrent-state size
(0.14 GiB) is an order of magnitude derived from configuration fields, not an execution
measurement; it is far too small to change any conclusion.

**Frozen.** The two files in `configs/` are unmodified copies of upstream configs,
retrieved 2026-08-14 and byte-identical to their sources on that date. They are frozen
deliberately: if upstream revises a config, the published numbers still correspond to
exactly these bytes. `pytest -m network` checks whether upstream has since changed.

## Integrity

A single SHA-256 chain seals the program, its tests, the frozen configs, the declared
numbers and the derived results:

```bash
python3 make_provenance.py --verify
```

Current `chain_hash` and the per-file digests are in
[`output/hash-chain.md`](output/hash-chain.md). Full replication — recompute, assert,
reseal, verify — is one command:

```bash
pip install -r requirements.txt
python3 run_all.py
```

Reproduction contract, including what can and cannot be re-run:
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Layout

```text
medidor.py                   the program (zero dependencies, single file)
run_all.py                   measure -> results.json -> tests -> provenance seal
make_provenance.py           build/verify the SHA-256 chain
configs/                     frozen upstream config.json files (the primary source)
data/published-numbers.json  every number the article publishes, with tolerances
tests/                       55 offline tests + 2 optional network checks against upstream
docs/algorithm.md            how the cache is computed, layer type by layer type
docs/findings.md             what the two models show, and why the gap differs
output/                      results.json, hash-chain.md, provenance.json
```

## Why the file is called `medidor.py`

*Medidor* is Portuguese for "meter" or "gauge". The article was published in Portuguese
first and prints `python3 medidor.py` in all five of its language editions, so the
filename is fixed by what readers already have in front of them. The program itself, its
documentation and its output are in English.

## Author

**Carlos Ulisses Flores** — CTO & Chief Researcher, Codex Hash Research Laboratory;
MSc AI candidate, American Global Tech University.

[![ORCID](https://img.shields.io/badge/ORCID-0000--0002--6034--7765-a6ce39.svg)](https://orcid.org/0000-0002-6034-7765)
[![Website](https://img.shields.io/badge/Website-ulissesflores.com-1f6feb.svg)](https://ulissesflores.com)
[![Lattes](https://img.shields.io/badge/Lattes-6905246706890561-2b7489.svg)](http://lattes.cnpq.br/6905246706890561)

## Citation

```bibtex
@software{flores_llm_memory_meter_2026,
  author  = {Flores, Carlos Ulisses},
  title   = {llm-memory-meter: measuring the real memory footprint of a local LLM
             from its official configuration},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/ulissesflores/llm-memory-meter}
}
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff) and
[`codemeta.json`](codemeta.json).

## License

Dual-licensed. Source code under **Apache-2.0**
([`LICENSES/Apache-2.0.txt`](LICENSES/Apache-2.0.txt)); documentation and declared data
under **CC BY 4.0** ([`LICENSES/CC-BY-4.0.txt`](LICENSES/CC-BY-4.0.txt)). The vendored
model configs in `configs/` remain under their upstream terms — see
[`NOTICE`](NOTICE).

## References

- [Qwen3.8-27B — model card and official configuration](https://huggingface.co/Qwen/Qwen3.8-27B)
- [Gemma 4 26B A4B — model card and official configuration](https://huggingface.co/google/gemma-4-26B-A4B-it)
- Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
- Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)
- Yang et al., [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)
- Gu & Dao, [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [llama.cpp — bits per weight for each quantization format](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [vLLM documentation on quantized KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/)

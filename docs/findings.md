# Findings

All figures below are produced by `python3 run_all.py` from the frozen configs in
`configs/`, and asserted by `tests/test_published_numbers.py`. Batch 1, BF16 cache unless
stated otherwise.

## 1. The overestimate is not a rounding error — it is an order of magnitude

| Model | Architecture | Overestimate at 256K |
|---|---|---:|
| Qwen3.8-27B | 48 linear + 16 full attention, GQA 6x | **19.8x** |
| Gemma 4 26B A4B | 25 sliding (window 1024) + 5 full, shared K=V | **8.2x** |

The error is not constant: it *grows* with context, because the formula's overcounted
layers scale linearly while the model's real growth is confined to a minority of layers.

| Context | Qwen3.8-27B | Gemma 4 26B A4B |
|---:|---:|---:|
| 1K | 6.2x | 2.4x |
| 8K | 15.6x | 6.3x |
| 32K | 18.7x | 7.7x |
| 128K | 19.7x | 8.1x |
| 256K | 19.8x | 8.2x |

At 1K tokens the formula is merely wrong; at long context it is useless. This matters
because long context is exactly when someone reaches for a memory estimate.

## 2. Three quarters of Qwen3.8-27B never grows

Of its 64 layers, 48 are recurrent. Together they hold **0.14 GiB — constant**, whether
the conversation is 1,000 tokens or 1,000,000. The remaining 16 full-attention layers do
all the growing, at 4 KiB per token each: 64 KiB per token across the model.

That single fact carries most of the gap. The viral formula charges all 64 layers the
growth rate of the 16.

## 3. The gap differs by model, for different reasons

Gemma 4's smaller error comes from a different mechanism entirely. Its sliding-window
layers *do* cache per token — they simply stop at 1024 tokens. Past the window, 25 of its
30 layers are flat. It also shares `K=V`, halving what its layers store.

Two models, two architectures, two different reasons the same formula fails. This is why
a calculator that does not ask about layer types cannot be right for both.

## 4. Real 4-bit quantization costs about 22% more than "4-bit"

| Format | Bytes/param | Qwen3.8-27B weights |
|---|---:|---:|
| BF16 | 2.0 | 51.75 GiB |
| INT8 | 1.0 | 25.87 GiB |
| 4-bit ideal | 0.5 | 12.94 GiB |
| **Q4_K_M (4.89 bpw)** | **0.61** | **15.82 GiB** |

The 2.88 GiB between ideal and real 4-bit is not a detail. On a 24 GB card it is most of
the margin that decides whether the model fits at all.

## 5. What actually fits on a 24 GB card

Q4_K_M weights take 15.82 GiB. After roughly 2 to 3 GB of engine overhead, about 5 GiB is
left for cache:

| Cache precision | Context that fits |
|---|---:|
| 16-bit | ~80,000 tokens |
| 8-bit | ~160,000 tokens |

Halving the cache precision doubles the conversation length. The article's practical
advice follows directly: **quantize the cache before shortening the conversation** — it
usually costs less quality than truncating context, and it buys twice the room.

The model's full 256K context would need about 8 GiB more than the card has. Its
advertised 1M context would need 64.1 GiB of cache alone, 80 GiB in total — a
workstation-class number, not a consumer one.

## 6. The name on the box understates the model

Qwen3.8-27B has **27,781,427,952** parameters — 2.89% more than "27B" suggests, worth
1.46 GiB at BF16. Model names round down. Memory does not.

## What would change these numbers

- **Upstream config revisions.** The frozen bytes are the ones measured on 2026-08-14.
  Run `pytest -m network` to check whether either model's config has changed since.
- **Running the model.** Every figure here is an architectural floor. Actual inference
  costs more; how much more depends on the engine and is not derived in this repository.
- **A different quantization mix.** Q4_K_M is one format among many; the program accepts
  any bytes-per-parameter you want to reason about.

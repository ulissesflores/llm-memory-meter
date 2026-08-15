# How the memory is computed

Two quantities compete for the same RAM, and they behave in opposite ways.

**Weights** are fixed. They occupy memory before a single token is processed, and their
size is known in advance: parameter count times bytes per parameter.

**The KV cache** grows. Each attention layer stores what it has already computed for
every token, so it does not have to recompute it. This is the part the viral formula
gets wrong, because it assumes every layer stores the same thing — which stopped being
true around 2024.

## The formula that circulates

```text
KV_cache = batch x context x layers x hidden_size x 2 x bytes_per_element
```

Every term is defensible, and the result is correct — for **pure Multi-Head Attention**.
`tests/test_formula_properties.py::test_viral_formula_is_exact_for_pure_mha` asserts that
this repository's implementation agrees with it exactly on an MHA model. The formula is
not a mistake; it is a description of an architecture that has largely been replaced.

It makes four assumptions, and modern models break them.

## Assumption 1 — every layer caches. Broken by hybrid architectures.

Models are no longer uniform stacks. `config.json` declares `layer_types`, and each type
has a different memory behaviour:

| Layer type | What it stores | Growth |
|---|---|---|
| `full_attention` | K and V for every token seen | Linear in context, unbounded |
| `sliding_attention` | K and V for the last `sliding_window` tokens | Linear until the window fills, then flat |
| `linear_attention` | One fixed-size recurrent state | **Constant** — independent of context |

Qwen3.8-27B is 48 `linear_attention` layers and 16 `full_attention` ones. Three quarters
of the model contributes a constant 0.14 GiB, no matter whether the conversation is 1,000
tokens or 1,000,000. The formula counts all 64 layers as growing.

Gemma 4 26B A4B is 25 `sliding_attention` layers (window 1024) and 5 `full_attention`
ones. Past 1024 tokens, five sixths of the model stops growing entirely.

A config with no `layer_types` field predates this distinction and is read as all full
attention — the historically correct default.

## Assumption 2 — the cache is `hidden_size` wide. Broken by Grouped-Query Attention.

`hidden_size` is the width of all *query* heads. The cache stores *keys and values*, and
since GQA those are shared across query heads:

```text
cache_width = num_key_value_heads x head_dim
```

For Qwen3.8-27B: 4 key/value heads times a head dimension of 256 gives **1024**, not the
5120 the formula assumes. The 24 query heads are served by 4 shared key/value heads — a
6x reduction, by design.

Some hybrid models give their full-attention layers a wider head dimension than their
sliding-window layers (`global_head_dim`). Gemma 4 does: 512 for its global layers, 256
for the windowed ones. The program applies each where it belongs.

## Assumption 3 — K and V are two tensors. Broken by shared K=V.

The factor of `2` counts one tensor for keys and one for values. When a model sets
`attention_k_eq_v`, the two are identical and only one is stored. Gemma 4 does this; the
factor is 1, and the formula is off by exactly 2x on those layers.

## Assumption 4 — a "4-bit" model uses 4 bits per weight. Broken by real quant formats.

llama.cpp's `Q4_K_M`, the most widely used 4-bit mix, averages **4.89 bits per weight** —
about 22% above the ideal — because scales and zero-points are stored alongside the
weights. For a 27.8B-parameter model that is the difference between 12.94 GiB and 15.82
GiB: nearly 3 GiB, which is often exactly the margin that decides whether a model fits on
a 24 GB card.

## What the program computes instead

For each layer type, the cache one token adds to one layer:

```text
kv_bytes_per_token_per_layer = kv_factor x num_key_value_heads x head_dim x bytes_per_element

  kv_factor = 1 when attention_k_eq_v, else 2
  head_dim  = global_head_dim on full-attention layers when the model declares one
```

Summed over the layers, each according to its own growth law:

```text
full_attention    : n_layers x batch x context                        x kv_bytes_per_token_per_layer
sliding_attention : n_layers x batch x min(context, sliding_window)   x kv_bytes_per_token_per_layer
linear_attention  : n_layers x batch x linear_state_bytes             (context does not appear)
```

where the recurrent state of one linear layer is

```text
linear_state_bytes = linear_num_value_heads x linear_key_head_dim x linear_value_head_dim x dtype_bytes
```

with `dtype_bytes` = 4 when `mamba_ssm_dtype` is `float32`, else 2.

Weights are the straightforward part:

```text
weight_bytes = total_parameters x bytes_per_parameter
```

using the exact parameter count reported by the Hub's `safetensors.total`, not the
rounded figure in the model's name. For Qwen3.8-27B the real count is 27,781,427,952 —
2.89% above "27B", which is 1.46 GiB of extra memory at BF16.

## Units

The program prints both, because vendors and operating systems disagree:

- **GB** = 10^9 bytes — what hardware is sold as.
- **GiB** = 2^30 bytes — what an operating system reports.

The gap is 7.37%, which is why a 1 TB drive shows up as 931 GB. The article and this
repository quote **GiB** for memory, since that is the number the machine will show you.

## Where this stops

The program reads architecture; it does not run models. It computes the floor the
architecture implies. A real inference engine adds activations, allocator padding, and
framework buffers on top — reported to be roughly 2 to 3 GB, but engine-dependent and not
derived here. Treat these figures as "no less than this", never "exactly this".

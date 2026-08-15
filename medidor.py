#!/usr/bin/env python3
"""Measure the real memory footprint of a local LLM from its official ``config.json``.

Why this exists
---------------
The formula that circulates in viral infographics,

.. code-block:: text

    KV_cache = batch x context x layers x hidden_size x 2 x bytes_per_element

is only correct for pure Multi-Head Attention. Every open model shipped in 2026 breaks
at least one of its assumptions:

- **Grouped-Query Attention (GQA)** — key/value heads are shared across query heads, so
  ``hidden_size`` is the wrong width. The cache is ``num_key_value_heads x head_dim``.
- **Sliding-window attention** — the cache stops growing once the window is full, so
  those layers are capped, not linear in context length.
- **Linear (recurrent) attention** — the layer keeps a fixed-size state that does not
  grow with context at all.
- **Shared K=V** — when a model stores one tensor instead of two, the factor of 2 is
  wrong.

This program reads the architecture the model actually declares and prints both numbers
side by side: what the viral formula predicts, and what the model really allocates.

Usage
-----
``python3 medidor.py``
    Run the two models discussed in the companion article, from the frozen configs
    shipped in ``configs/``. No network access.

``python3 medidor.py --repo Qwen/Qwen3.8-27B``
    Download ``config.json`` and the parameter count from the Hugging Face Hub.

``python3 medidor.py configs/Qwen3.8-27B.config.json --params 27781427952``
    Read a local config file; supply the parameter count yourself.

Options
-------
``--batch N``
    Number of concurrent sequences (default 1). The cache scales linearly with it.

``--bytes-cache N``
    Bytes per cached element: 2 for BF16/FP16 (default), 1 for an 8-bit quantized cache.

Notes
-----
Every number is a *theoretical floor* derived from the declared architecture. The model
is never executed, so runtime overhead (allocator padding, activations, framework
buffers) is not included and always costs somewhat more in practice.

Companion article: https://ulissesflores.com/artigos/memoria-llm-local
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import urllib.request
from pathlib import Path

GiB = 2**30
"""Bytes in a gibibyte — what an operating system calls a "GB"."""

GB = 1e9
"""Bytes in a gigabyte — what a hardware vendor calls a "GB"."""

ROOT = Path(__file__).resolve().parent
"""Repository root, used to locate the frozen configs in ``configs/``."""

GROWS_UNBOUNDED = {"full_attention"}
"""Layer types whose cache grows with every token, forever."""

GROWS_TO_WINDOW = {"sliding_attention"}
"""Layer types whose cache grows only until the sliding window is full."""

HF_RAW = "https://huggingface.co/{repo}/raw/main/config.json"
"""Template for the canonical raw ``config.json`` URL on the Hugging Face Hub."""

HF_API = "https://huggingface.co/api/models/{repo}"
"""Template for the Hub metadata endpoint that reports the safetensors parameter count."""

CONTEXTS = (1024, 8192, 32768, 131072, 262144)
"""Context lengths reported in the comparison table, in tokens."""

WEIGHT_FORMATS = (
    ("BF16", 2.0),
    ("INT8", 1.0),
    ("4-bit ideal", 0.5),
    ("Q4_K_M (4.89 bpw)", 4.89 / 8),
)
"""Weight formats and their bytes per parameter.

``Q4_K_M`` is llama.cpp's most common 4-bit mix; at 4.89 bits per weight it costs about
22% more than the ideal 4 bits, because scales and zero-points are stored too.
"""


def fetch_json(url: str, timeout: int = 30) -> dict:
    """Download and parse a JSON document.

    Parameters
    ----------
    url : str
        Absolute URL returning a JSON body.
    timeout : int, optional
        Socket timeout in seconds, by default 30.

    Returns
    -------
    dict
        The parsed document.
    """
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read())


def load_config(repo: str | None, path: str | None) -> tuple[dict, int | None, str]:
    """Load a model config from the Hub or from disk.

    Parameters
    ----------
    repo : str or None
        Hugging Face repository id, e.g. ``"Qwen/Qwen3.8-27B"``. Takes precedence.
    path : str or None
        Path to a local ``config.json`` when ``repo`` is not given.

    Returns
    -------
    tuple of (dict, int or None, str)
        The config, the total parameter count when the Hub reports one, and a label
        for the report header.
    """
    if repo:
        config = fetch_json(HF_RAW.format(repo=repo))
        try:
            metadata = fetch_json(HF_API.format(repo=repo))
            n_params = (metadata.get("safetensors") or {}).get("total")
        except Exception:  # noqa: BLE001 — the parameter count is optional, the config is not
            n_params = None
        return config, n_params, repo
    return json.loads(Path(path).read_text()), None, Path(path).stem


def architecture(config: dict) -> dict:
    """Extract the fields that decide memory cost from a raw config.

    Flattens the ``text_config`` nesting used by multimodal releases (Gemma, Qwen-VL)
    and fills in the fields older configs leave implicit: a model with no ``layer_types``
    is all full attention, and a missing ``head_dim`` is ``hidden_size / num_attention_heads``.

    Parameters
    ----------
    config : dict
        A parsed ``config.json``.

    Returns
    -------
    dict
        Normalized architecture description consumed by the cost functions.
    """
    text = config.get("text_config", config)
    layer_types = text.get("layer_types") or []
    n_layers = text.get("num_hidden_layers") or len(layer_types)
    if not layer_types:
        layer_types = ["full_attention"] * n_layers

    n_kv = text.get("num_key_value_heads") or text.get("num_attention_heads")
    head_dim = text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"]

    return {
        "layers": n_layers,
        "layer_counts": collections.Counter(layer_types),
        "hidden": text["hidden_size"],
        "n_heads": text.get("num_attention_heads"),
        "n_kv": n_kv,
        "head_dim": head_dim,
        "global_head_dim": text.get("global_head_dim"),
        "window": text.get("sliding_window"),
        # Gemma 4 stores a single tensor when K and V are identical, halving the cache.
        "k_eq_v": bool(text.get("attention_k_eq_v")),
        "max_context": text.get("max_position_embeddings"),
        "linear": {
            "n_value_heads": text.get("linear_num_value_heads"),
            "k_dim": text.get("linear_key_head_dim"),
            "v_dim": text.get("linear_value_head_dim"),
            "dtype_bytes": 4 if text.get("mamba_ssm_dtype") == "float32" else 2,
        },
    }


def kv_bytes_per_token_per_layer(
    arch: dict, bytes_cache: int, *, global_layer: bool = False
) -> int:
    """Return the cache bytes one token adds to one attention layer.

    This is the number the viral formula gets wrong: it uses ``hidden_size``, the width
    of all *query* heads, where the cache is only as wide as the *key/value* heads.

    Parameters
    ----------
    arch : dict
        Output of :func:`architecture`.
    bytes_cache : int
        Bytes per cached element (2 for BF16, 1 for an 8-bit cache).
    global_layer : bool, optional
        True for full-attention layers, which some hybrid models give a wider head
        dimension than their sliding-window layers, by default False.

    Returns
    -------
    int
        Bytes added per token, per layer.
    """
    head_dim = (
        arch["global_head_dim"] if (global_layer and arch["global_head_dim"]) else arch["head_dim"]
    )
    kv_factor = 1 if arch["k_eq_v"] else 2
    return kv_factor * arch["n_kv"] * head_dim * bytes_cache


def linear_state_bytes(arch: dict) -> int:
    """Return the recurrent state size of one linear-attention layer.

    Constant by construction: a recurrent layer summarizes everything it has seen into
    a fixed-size state, so it costs the same at 1K tokens and at 1M.

    Parameters
    ----------
    arch : dict
        Output of :func:`architecture`.

    Returns
    -------
    int
        Bytes per layer, or 0 when the model has no linear-attention layers.
    """
    linear = arch["linear"]
    if not linear["n_value_heads"]:
        return 0
    return linear["n_value_heads"] * linear["k_dim"] * linear["v_dim"] * linear["dtype_bytes"]


def real_cache_bytes(arch: dict, context: int, batch: int, bytes_cache: int) -> int:
    """Return the cache the model actually allocates, layer type by layer type.

    Parameters
    ----------
    arch : dict
        Output of :func:`architecture`.
    context : int
        Context length in tokens.
    batch : int
        Number of concurrent sequences.
    bytes_cache : int
        Bytes per cached element.

    Returns
    -------
    int
        Total cache bytes.
    """
    total = 0
    for layer_type, count in arch["layer_counts"].items():
        if layer_type in GROWS_UNBOUNDED:
            per_token = kv_bytes_per_token_per_layer(arch, bytes_cache, global_layer=True)
            total += count * batch * context * per_token
        elif layer_type in GROWS_TO_WINDOW:
            effective = min(context, arch["window"] or context)
            total += count * batch * effective * kv_bytes_per_token_per_layer(arch, bytes_cache)
        else:
            total += count * batch * linear_state_bytes(arch)
    return total


def viral_formula_bytes(arch: dict, context: int, batch: int, bytes_cache: int) -> int:
    """Return what the viral formula predicts, applied literally.

    ``batch x context x layers x hidden_size x 2 x bytes`` — every layer counted as full
    attention, at full model width.

    Parameters
    ----------
    arch : dict
        Output of :func:`architecture`.
    context : int
        Context length in tokens.
    batch : int
        Number of concurrent sequences.
    bytes_cache : int
        Bytes per cached element.

    Returns
    -------
    int
        Predicted cache bytes.
    """
    return batch * context * arch["layers"] * arch["hidden"] * 2 * bytes_cache


def measure(
    config: dict, n_params: int | None = None, batch: int = 1, bytes_cache: int = 2
) -> dict:
    """Compute every published number for one model, as plain data.

    This is the function the tests and ``run_all.py`` call; :func:`report` only formats
    what this returns.

    Parameters
    ----------
    config : dict
        A parsed ``config.json``.
    n_params : int or None, optional
        Total parameter count, used for the weight table, by default None.
    batch : int, optional
        Number of concurrent sequences, by default 1.
    bytes_cache : int, optional
        Bytes per cached element, by default 2 (BF16).

    Returns
    -------
    dict
        ``{"architecture", "weights", "linear_state", "cache"}`` with byte counts and
        the GiB values the article publishes.
    """
    arch = architecture(config)

    weights = {}
    if n_params:
        for name, bytes_per_param in WEIGHT_FORMATS:
            total = n_params * bytes_per_param
            weights[name] = {"bytes": total, "GB": total / GB, "GiB": total / GiB}

    per_layer = linear_state_bytes(arch)
    n_linear = sum(
        count
        for layer_type, count in arch["layer_counts"].items()
        if layer_type not in GROWS_UNBOUNDED | GROWS_TO_WINDOW
    )

    cache = {}
    for context in CONTEXTS:
        if arch["max_context"] and context > arch["max_context"]:
            continue
        viral = viral_formula_bytes(arch, context, batch, bytes_cache)
        real = real_cache_bytes(arch, context, batch, bytes_cache)
        cache[context] = {
            "viral_GiB": viral / GiB,
            "real_GiB": real / GiB,
            "overestimate_x": viral / real,
        }

    return {
        "architecture": {
            "layers": arch["layers"],
            "layer_counts": dict(arch["layer_counts"]),
            "hidden": arch["hidden"],
            "n_heads": arch["n_heads"],
            "n_kv": arch["n_kv"],
            "head_dim": arch["head_dim"],
            "global_head_dim": arch["global_head_dim"],
            "window": arch["window"],
            "k_eq_v": arch["k_eq_v"],
            "max_context": arch["max_context"],
        },
        "weights": weights,
        "linear_state": {
            "bytes_per_layer": per_layer,
            "n_layers": n_linear,
            "total_GiB": n_linear * per_layer / GiB,
        },
        "cache": cache,
        "batch": batch,
        "bytes_cache": bytes_cache,
    }


def report(measurement: dict, label: str) -> None:
    """Print one model's measurement as a human-readable report.

    Parameters
    ----------
    measurement : dict
        Output of :func:`measure`.
    label : str
        Header shown above the report.
    """
    arch = measurement["architecture"]
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    print(f"layers: {arch['layers']} -> {arch['layer_counts']}")

    line = (
        f"hidden {arch['hidden']} | heads {arch['n_heads']} / kv {arch['n_kv']} "
        f"(GQA {arch['n_heads'] // arch['n_kv']}x) | head_dim {arch['head_dim']}"
    )
    if arch["global_head_dim"]:
        line += f" (global {arch['global_head_dim']})"
    if arch["k_eq_v"]:
        line += " | shared K=V"
    if arch["window"]:
        line += f" | window {arch['window']}"
    print(line)

    if measurement["weights"]:
        print("\nWEIGHTS:")
        for name, value in measurement["weights"].items():
            print(f"  {name:20} {value['GB']:7.2f} GB | {value['GiB']:7.2f} GiB")

    state = measurement["linear_state"]
    if state["bytes_per_layer"]:
        print(
            f"\nlinear state: {state['bytes_per_layer'] / 2**20:.1f} MiB/layer "
            f"x {state['n_layers']} = {state['total_GiB']:.2f} GiB CONSTANT "
            f"(does not grow with context)"
        )

    print(f"\ncache at batch={measurement['batch']}, {measurement['bytes_cache']} byte(s)/element:")
    print(f"{'context':>10} | {'viral formula':>16} | {'real':>12} | {'error':>6}")
    print("-" * 54)
    for context, value in measurement["cache"].items():
        print(
            f"{context // 1024:>9}K | {value['viral_GiB']:12.2f} GiB | "
            f"{value['real_GiB']:8.2f} GiB | {value['overestimate_x']:5.1f}x"
        )


def article_models() -> list[tuple[Path, int | None, str]]:
    """Return the frozen configs of the two models discussed in the article.

    Returns
    -------
    list of (Path, int or None, str)
        Config path, parameter count, and report label for each model.
    """
    return [
        (
            ROOT / "configs" / "Qwen3.8-27B.config.json",
            27_781_427_952,
            "Qwen3.8-27B (the article's worked example)",
        ),
        (
            ROOT / "configs" / "gemma-4-26B-A4B-it.config.json",
            None,
            "Gemma 4 26B A4B (the model in the viral infographic)",
        ),
    ]


def main() -> int:
    """Parse arguments, measure, and print.

    Returns
    -------
    int
        0 on success, 1 when the frozen configs are missing.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", nargs="?", help="path to a local config.json")
    parser.add_argument("--repo", help="Hugging Face repo id (downloads config + parameter count)")
    parser.add_argument("--params", type=int, help="total parameters, when reading a local config")
    parser.add_argument("--batch", type=int, default=1, help="concurrent sequences (default 1)")
    parser.add_argument(
        "--bytes-cache",
        type=int,
        default=2,
        help="bytes per cached element: 2=BF16 (default), 1=8-bit quantized cache",
    )
    args = parser.parse_args()

    if not args.config and not args.repo:
        found = False
        for path, params, label in article_models():
            if path.exists():
                report(
                    measure(json.loads(path.read_text()), params, args.batch, args.bytes_cache),
                    label,
                )
                found = True
            else:
                print(f"[missing] {path}", file=sys.stderr)
        return 0 if found else 1

    config, hub_params, label = load_config(args.repo, args.config)
    report(measure(config, args.params or hub_params, args.batch, args.bytes_cache), label)
    return 0


if __name__ == "__main__":
    sys.exit(main())

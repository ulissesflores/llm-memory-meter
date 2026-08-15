"""Check the structural properties that make the viral formula wrong.

These tests do not depend on any particular model: they assert the behaviour that any
correct implementation must have. Each one corresponds to a reason the one-line formula
breaks on a 2026 architecture.
"""

import json
from pathlib import Path

import pytest

import medidor

ROOT = Path(__file__).resolve().parent.parent


def load(name: str) -> dict:
    """Load a frozen config by file name.

    Parameters
    ----------
    name : str
        File name inside ``configs/``.

    Returns
    -------
    dict
        The parsed config.
    """
    return json.loads((ROOT / "configs" / name).read_text())


MHA_CONFIG = {
    "num_hidden_layers": 4,
    "hidden_size": 512,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "head_dim": 64,
}
"""A pure Multi-Head Attention model: no GQA, no window, no linear layers.

This is the only architecture the viral formula was ever correct for, so it is the
control case — the one configuration where both numbers must agree exactly.
"""


def test_viral_formula_is_exact_for_pure_mha() -> None:
    """On pure MHA the two formulas agree — the formula is not wrong, it is outdated."""
    arch = medidor.architecture(MHA_CONFIG)
    for context in (1024, 8192, 32768):
        viral = medidor.viral_formula_bytes(arch, context, 1, 2)
        real = medidor.real_cache_bytes(arch, context, 1, 2)
        assert viral == real


def test_gqa_shrinks_the_cache_by_the_head_ratio() -> None:
    """Sharing 8 query heads over 2 key/value heads divides the cache by 4."""
    mha = medidor.architecture(MHA_CONFIG)
    gqa = medidor.architecture({**MHA_CONFIG, "num_key_value_heads": 2})
    assert medidor.real_cache_bytes(mha, 8192, 1, 2) == 4 * medidor.real_cache_bytes(
        gqa, 8192, 1, 2
    )


def test_viral_formula_never_underestimates() -> None:
    """For every model and context here, the viral figure is an upper bound."""
    for name in ("Qwen3.8-27B.config.json", "gemma-4-26B-A4B-it.config.json"):
        arch = medidor.architecture(load(name))
        for context in medidor.CONTEXTS:
            viral = medidor.viral_formula_bytes(arch, context, 1, 2)
            real = medidor.real_cache_bytes(arch, context, 1, 2)
            assert viral >= real, f"{name} at {context}"


def test_linear_layers_do_not_grow_with_context() -> None:
    """A recurrent layer costs the same at 1K tokens and at 1M — that is the point."""
    arch = medidor.architecture(load("Qwen3.8-27B.config.json"))
    linear_only = {
        **arch,
        "layer_counts": {"linear_attention": arch["layer_counts"]["linear_attention"]},
    }
    sizes = {medidor.real_cache_bytes(linear_only, ctx, 1, 2) for ctx in (1024, 262144, 1048576)}
    assert len(sizes) == 1
    assert sizes.pop() > 0


def test_sliding_window_saturates_at_the_window() -> None:
    """Once the window is full, more context costs a sliding layer nothing."""
    arch = medidor.architecture(load("gemma-4-26B-A4B-it.config.json"))
    window = arch["window"]
    sliding_only = {
        **arch,
        "layer_counts": {"sliding_attention": arch["layer_counts"]["sliding_attention"]},
    }
    at_window = medidor.real_cache_bytes(sliding_only, window, 1, 2)
    beyond = medidor.real_cache_bytes(sliding_only, window * 64, 1, 2)
    assert at_window == beyond
    assert medidor.real_cache_bytes(sliding_only, window // 2, 1, 2) < at_window


def test_shared_kv_halves_the_cache() -> None:
    """When K and V are the same tensor, the factor of 2 in the formula is wrong."""
    base = {**MHA_CONFIG, "num_key_value_heads": 2}
    separate = medidor.architecture(base)
    shared = medidor.architecture({**base, "attention_k_eq_v": True})
    assert medidor.real_cache_bytes(separate, 8192, 1, 2) == 2 * medidor.real_cache_bytes(
        shared, 8192, 1, 2
    )


def test_quantized_cache_halves_the_growing_part() -> None:
    """An 8-bit cache costs half of a 16-bit one, per token."""
    arch = medidor.architecture(load("Qwen3.8-27B.config.json"))
    assert medidor.kv_bytes_per_token_per_layer(
        arch, 1
    ) * 2 == medidor.kv_bytes_per_token_per_layer(arch, 2)


def test_cache_scales_linearly_with_batch() -> None:
    """Two concurrent sequences cost twice the cache of one."""
    arch = medidor.architecture(load("Qwen3.8-27B.config.json"))
    one = medidor.real_cache_bytes(arch, 32768, 1, 2)
    four = medidor.real_cache_bytes(arch, 32768, 4, 2)
    assert four == 4 * one


def test_missing_layer_types_defaults_to_full_attention() -> None:
    """A config from before hybrid models is read as all full attention."""
    arch = medidor.architecture(MHA_CONFIG)
    assert arch["layer_counts"] == {"full_attention": MHA_CONFIG["num_hidden_layers"]}


def test_head_dim_falls_back_to_hidden_over_heads() -> None:
    """Configs that omit head_dim get the classic hidden_size / num_heads."""
    config = {k: v for k, v in MHA_CONFIG.items() if k != "head_dim"}
    arch = medidor.architecture(config)
    assert arch["head_dim"] == config["hidden_size"] // config["num_attention_heads"]


def test_nested_text_config_is_flattened() -> None:
    """Multimodal releases nest the language model under text_config."""
    nested = medidor.architecture({"text_config": MHA_CONFIG, "vision_config": {"hidden_size": 1}})
    assert nested["hidden"] == MHA_CONFIG["hidden_size"]
    assert nested["layers"] == MHA_CONFIG["num_hidden_layers"]


def test_measure_is_deterministic() -> None:
    """The same config yields byte-identical results on every run — no randomness."""
    config = load("Qwen3.8-27B.config.json")
    first = medidor.measure(config, n_params=27_781_427_952)
    second = medidor.measure(config, n_params=27_781_427_952)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_contexts_above_the_model_maximum_are_skipped() -> None:
    """The default table stops at what the model actually supports."""
    arch = medidor.architecture(load("gemma-4-26B-A4B-it.config.json"))
    measurement = medidor.measure(load("gemma-4-26B-A4B-it.config.json"))
    if arch["max_context"]:
        assert all(ctx <= arch["max_context"] for ctx in measurement["cache"])


@pytest.mark.parametrize("bytes_cache", [1, 2])
def test_zero_context_costs_only_the_fixed_state(bytes_cache: int) -> None:
    """With no tokens, only the constant recurrent state is allocated."""
    arch = medidor.architecture(load("Qwen3.8-27B.config.json"))
    expected = arch["layer_counts"]["linear_attention"] * medidor.linear_state_bytes(arch)
    assert medidor.real_cache_bytes(arch, 0, 1, bytes_cache) == expected

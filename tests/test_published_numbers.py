"""Lock every number the companion article publishes to what this program computes.

Each test reads the claim from ``data/published-numbers.json`` and recomputes it from the
frozen ``config.json`` files in ``configs/``. If the code changes and a published figure
moves, CI turns red — the article and the program cannot drift apart silently.

No network access: the Hub download path is exercised separately and is skipped by default.
"""

import json
from pathlib import Path

import pytest

import medidor

ROOT = Path(__file__).resolve().parent.parent
PUBLISHED = json.loads((ROOT / "data" / "published-numbers.json").read_text())

QWEN = PUBLISHED["Qwen3.8-27B"]
GEMMA = PUBLISHED["gemma-4-26B-A4B-it"]


def load(name: str) -> dict:
    """Load a frozen config by file stem.

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


@pytest.fixture(scope="module")
def qwen() -> dict:
    """Measure Qwen3.8-27B once for the whole module.

    Returns
    -------
    dict
        Output of :func:`medidor.measure`.
    """
    return medidor.measure(
        load("Qwen3.8-27B.config.json"),
        n_params=QWEN["architecture"]["total_params"],
    )


@pytest.fixture(scope="module")
def gemma() -> dict:
    """Measure Gemma 4 26B A4B once for the whole module.

    Returns
    -------
    dict
        Output of :func:`medidor.measure`.
    """
    return medidor.measure(load("gemma-4-26B-A4B-it.config.json"))


# --------------------------------------------------------------------------- architecture


def test_qwen_architecture(qwen: dict) -> None:
    """The article's architecture claims match the official config."""
    arch = qwen["architecture"]
    published = QWEN["architecture"]
    assert arch["layers"] == published["layers"]
    assert arch["layer_counts"]["linear_attention"] == published["linear_attention_layers"]
    assert arch["layer_counts"]["full_attention"] == published["full_attention_layers"]
    assert arch["hidden"] == published["hidden_size"]
    assert arch["n_heads"] == published["attention_heads"]
    assert arch["n_kv"] == published["key_value_heads"]
    assert arch["head_dim"] == published["head_dim"]
    assert arch["max_context"] == published["max_context"]


def test_qwen_gqa_ratio_and_kv_width(qwen: dict) -> None:
    """GQA shares 24 query heads over 4 key/value heads, giving a 1024-wide cache."""
    arch = qwen["architecture"]
    assert arch["n_heads"] // arch["n_kv"] == QWEN["architecture"]["gqa_ratio"]
    assert arch["n_kv"] * arch["head_dim"] == QWEN["architecture"]["kv_width"]


def test_gemma_architecture(gemma: dict) -> None:
    """Gemma 4 26B A4B is 25 sliding-window layers plus 5 full-attention ones."""
    arch = gemma["architecture"]
    published = GEMMA["architecture"]
    assert arch["layers"] == published["layers"]
    assert arch["layer_counts"]["sliding_attention"] == published["sliding_attention_layers"]
    assert arch["layer_counts"]["full_attention"] == published["full_attention_layers"]
    assert arch["window"] == published["sliding_window"]
    assert arch["global_head_dim"] == published["global_head_dim"]
    assert arch["k_eq_v"] is published["shared_k_eq_v"]


# --------------------------------------------------------------------------------- weights


@pytest.mark.parametrize("fmt", ["BF16", "INT8", "4-bit ideal", "Q4_K_M"])
def test_weight_footprint(qwen: dict, fmt: str) -> None:
    """Weight memory in each format matches the published figure."""
    published = QWEN["weights_GiB"]
    key = "Q4_K_M (4.89 bpw)" if fmt == "Q4_K_M" else fmt
    assert qwen["weights"][key]["GiB"] == pytest.approx(published[fmt], abs=published["tolerance"])


def test_q4km_costs_22_percent_more_than_ideal_4bit() -> None:
    """Q4_K_M's 4.89 bits per weight is ~22% above the ideal 4 bits."""
    quant = QWEN["quantization"]
    overhead_pct = (quant["Q4_K_M_bits_per_weight"] / 4.0 - 1) * 100
    assert overhead_pct == pytest.approx(quant["overhead_vs_ideal_4bit_pct"], abs=0.01)
    assert quant["Q4_K_M_bits_per_weight"] / 8 == pytest.approx(
        quant["Q4_K_M_bytes_per_param"], abs=0.005
    )


def test_exact_parameter_count_beats_the_name(qwen: dict) -> None:
    """The name "27B" understates the real count by 2.89%, worth 1.46 GiB at BF16."""
    gap = QWEN["naming_gap"]
    real = QWEN["architecture"]["total_params"]
    assert (real / gap["nominal_params"] - 1) * 100 == pytest.approx(
        gap["pct_more_than_nominal"], abs=gap["tolerance"]
    )
    extra_gib = (real - gap["nominal_params"]) * 2 / medidor.GiB
    assert extra_gib == pytest.approx(gap["extra_GiB_at_bf16"], abs=gap["tolerance"])


# ---------------------------------------------------------------------------- linear state


def test_linear_state_is_constant_and_small(qwen: dict) -> None:
    """The 48 recurrent layers cost 3 MiB each — 0.14 GiB that never grows."""
    published = QWEN["linear_state"]
    state = qwen["linear_state"]
    assert state["n_layers"] == QWEN["architecture"]["linear_attention_layers"]
    assert state["bytes_per_layer"] / 2**20 == pytest.approx(
        published["MiB_per_layer"], abs=published["tolerance"]
    )
    assert state["total_GiB"] == pytest.approx(published["total_GiB"], abs=published["tolerance"])


# ----------------------------------------------------------------------------------- cache


def test_cache_bytes_per_token(qwen: dict) -> None:
    """One token adds 4 KiB per full-attention layer, 64 KiB across the model."""
    published = QWEN["cache_per_token"]
    arch = medidor.architecture(load("Qwen3.8-27B.config.json"))
    per_layer = medidor.kv_bytes_per_token_per_layer(arch, bytes_cache=2, global_layer=True)
    assert per_layer / 1024 == pytest.approx(published["KiB_per_full_attention_layer"])
    n_full = QWEN["architecture"]["full_attention_layers"]
    assert per_layer * n_full / 1024 == pytest.approx(published["KiB_total_all_layers"])


@pytest.mark.parametrize("context", ["1024", "8192", "32768", "131072", "262144"])
def test_qwen_cache_size(qwen: dict, context: str) -> None:
    """Cache size at each published context length."""
    published = QWEN["cache_GiB_bf16_batch1"]
    assert qwen["cache"][int(context)]["real_GiB"] == pytest.approx(
        published[context], abs=published["tolerance"]
    )


@pytest.mark.parametrize("context", ["1024", "8192", "32768", "131072", "262144"])
def test_qwen_total_footprint(qwen: dict, context: str) -> None:
    """Quantized weights plus cache — the number that decides if the model fits."""
    published = QWEN["total_GiB_q4km_plus_cache"]
    total = qwen["weights"]["Q4_K_M (4.89 bpw)"]["GiB"] + qwen["cache"][int(context)]["real_GiB"]
    assert total == pytest.approx(published[context], abs=published["tolerance"])


@pytest.mark.parametrize("context", ["1024", "8192", "32768", "131072", "262144"])
def test_qwen_viral_formula_error(qwen: dict, context: str) -> None:
    """How many times the viral formula overestimates, at each context length."""
    published = QWEN["viral_formula_overestimate_x"]
    assert qwen["cache"][int(context)]["overestimate_x"] == pytest.approx(
        published[context], abs=published["tolerance"]
    )


def test_headline_error_is_about_20x(qwen: dict) -> None:
    """The article's headline number: ~20x at the model's maximum context."""
    assert qwen["cache"][262144]["overestimate_x"] == pytest.approx(19.8, abs=0.2)


@pytest.mark.parametrize("context", ["1024", "8192", "32768", "131072", "262144"])
def test_gemma_viral_formula_error(gemma: dict, context: str) -> None:
    """The infographic's own model is overestimated 2.4x to 8.2x."""
    published = GEMMA["viral_formula_overestimate_x"]
    assert gemma["cache"][int(context)]["overestimate_x"] == pytest.approx(
        published[context], abs=published["tolerance"]
    )


# ------------------------------------------------------------------------------- scenarios


def test_million_token_context() -> None:
    """The advertised 1M context would cost 64.1 GiB of cache, 80 GiB in total."""
    published = QWEN["million_token_scenario"]
    arch = medidor.architecture(load("Qwen3.8-27B.config.json"))
    cache_gib = medidor.real_cache_bytes(arch, published["context"], 1, 2) / medidor.GiB
    assert cache_gib == pytest.approx(published["cache_GiB"], abs=published["tolerance"])

    weights_gib = QWEN["architecture"]["total_params"] * (4.89 / 8) / medidor.GiB
    assert weights_gib + cache_gib == pytest.approx(
        published["total_GiB"], abs=published["tolerance"]
    )


@pytest.mark.parametrize(
    ("bytes_cache", "key"), [(2, "max_context_16bit_cache"), (1, "max_context_8bit_cache")]
)
def test_rtx_3090_context_budget(bytes_cache: int, key: str) -> None:
    """With 5 GiB free, a 16-bit cache buys ~80K tokens and an 8-bit one ~160K.

    Halving the bytes per cached element doubles the context that fits — the article's
    practical advice, checked against the arithmetic.
    """
    published = QWEN["rtx_3090_scenario"]
    arch = medidor.architecture(load("Qwen3.8-27B.config.json"))
    budget = published["free_after_weights_GiB"] * medidor.GiB

    per_token = (
        medidor.kv_bytes_per_token_per_layer(arch, bytes_cache, global_layer=True)
        * arch["layer_counts"]["full_attention"]
    )
    fixed = arch["layer_counts"]["linear_attention"] * medidor.linear_state_bytes(arch)
    fits = int((budget - fixed) // per_token)

    assert fits == pytest.approx(published[key], abs=published["tolerance_tokens"])


# ------------------------------------------------------------------------ unit conversions


def test_unit_conversions() -> None:
    """The GB/GiB gap and the 256K-is-not-256000 gap the article explains."""
    published = PUBLISHED["unit_conversions"]
    tolerance = published["tolerance"]
    assert (medidor.GiB / medidor.GB - 1) * 100 == pytest.approx(
        published["GiB_over_GB_pct"], abs=tolerance
    )
    assert (262144 / 256000 - 1) * 100 == pytest.approx(
        published["tokens_256K_vs_256000_pct"], abs=tolerance
    )
    assert 1e12 / medidor.GiB == pytest.approx(published["one_TB_drive_shown_as_GB"], abs=1)

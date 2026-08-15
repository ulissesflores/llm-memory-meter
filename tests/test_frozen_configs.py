"""Verify the frozen evidence: the vendored configs are the exact bytes that were measured.

The configs in ``configs/`` are the article's primary source. Their SHA-256 digests are
recorded in ``data/published-numbers.json`` and asserted here, so a config cannot be
edited — accidentally or otherwise — without the suite failing.

The optional test at the end compares them against the live Hugging Face Hub. It is
skipped by default: CI must not depend on the network, and upstream is free to update a
config at any time. A mismatch there is information, not a defect — run it with
``pytest -m network`` when you want to know whether upstream has moved.
"""

import hashlib
import json
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PUBLISHED = json.loads((ROOT / "data" / "published-numbers.json").read_text())

FROZEN = {
    "Qwen3.8-27B.config.json": ("Qwen3.8-27B", "Qwen/Qwen3.8-27B"),
    "gemma-4-26B-A4B-it.config.json": ("gemma-4-26B-A4B-it", "google/gemma-4-26B-A4B-it"),
}
"""Vendored file -> (key in published-numbers.json, Hugging Face repo id)."""


@pytest.mark.parametrize("filename", sorted(FROZEN))
def test_frozen_config_digest(filename: str) -> None:
    """Each vendored config still hashes to the digest recorded when it was measured."""
    key, _ = FROZEN[filename]
    digest = hashlib.sha256((ROOT / "configs" / filename).read_bytes()).hexdigest()
    assert digest == PUBLISHED[key]["config_sha256"]


@pytest.mark.parametrize("filename", sorted(FROZEN))
def test_frozen_config_is_valid_json_with_the_fields_we_rely_on(filename: str) -> None:
    """A config without these fields would make the measurement meaningless."""
    config = json.loads((ROOT / "configs" / filename).read_text())
    text = config.get("text_config", config)
    assert text["hidden_size"] > 0
    assert text["num_attention_heads"] > 0
    assert text.get("num_hidden_layers") or text.get("layer_types")


@pytest.mark.network
@pytest.mark.parametrize("filename", sorted(FROZEN))
def test_frozen_config_matches_upstream(filename: str) -> None:
    """Optional: the vendored bytes still match the model's live config on the Hub.

    Skipped unless run with ``pytest -m network``. Upstream changing its config does not
    invalidate the published numbers — it means the article measured an earlier revision,
    which is exactly why the bytes are frozen here.
    """
    _, repo = FROZEN[filename]
    url = f"https://huggingface.co/{repo}/raw/main/config.json"
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        upstream = response.read()
    assert (ROOT / "configs" / filename).read_bytes() == upstream, (
        f"{repo} changed its config.json upstream; the frozen copy is the measured revision"
    )

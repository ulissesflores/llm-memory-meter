# Provenance hash chain

**chain_hash:** `e052b61901f613cb17034e5d7ab6ab7a16dc8573fa51a1d1215088d236fa3505`

Recompute and compare with `python make_provenance.py --verify`.

## Hashed — the published numbers depend on these bytes

| File | SHA-256 |
|---|---|
| `configs/Qwen3.8-27B.config.json` | `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab` |
| `configs/gemma-4-26B-A4B-it.config.json` | `ed0c1eb3633de771906e9ba004a44cc5635bcc06ee2062077c3d2e88a50707d3` |
| `data/published-numbers.json` | `a30da29ea46aed84cad49166fe4c47f3096cb4c55d3d5640d20768e1babde7fc` |
| `medidor.py` | `bf1c6ce623edfabaf7a0ed6880b9367776d1fbc27f9a7473b6aa94f52b1d3272` |
| `output/results.json` | `d08b6b3d7ab39f4674e6f7c53e8c88c5b2f15bb92cf6f722d1f539616680e753` |
| `run_all.py` | `f4f42e72e829a34dcb480dec5981302c1646a858c5ce4b1b779526ebc31334e8` |
| `tests/test_formula_properties.py` | `6e8ad820b58157fbdf52cc24cde09911ac1430d7828e4b9c5c8a80c3848c4aa2` |
| `tests/test_frozen_configs.py` | `0b5b2937daf93abfff587aa3d9148fdc774af29cbebd1a898271f5317fe6181f` |
| `tests/test_published_numbers.py` | `b9e110f31bbbaf27b262f6381bd1b11deeab826462f58bc32c05bb74d9b25b51` |

## Informational — NOT hashed

- Generated: 2026-08-14
- Python: 3.14.6 on Darwin arm64
- Documentation, `requirements.lock` and CI configuration are excluded so the
  chain survives machine and toolchain changes. What is sealed is the program,
  the frozen model configs, the published numbers and the derived results.

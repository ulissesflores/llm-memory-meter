"""Build (or verify) the SHA-256 provenance chain of this repository.

The chain covers what the published claim depends on — the program, its tests, the frozen
model configs, the declared numbers and the derived results — and deliberately excludes
what varies per machine (interpreter version, platform, lockfile). One ``chain_hash``
seals the set: if any hashed byte changes, the hash changes.

Usage
-----
``python make_provenance.py``
    Build ``output/provenance.json`` and ``output/hash-chain.md``.

``python make_provenance.py --verify``
    Recompute the chain and compare it against the stored seal.
"""

import hashlib
import json
import platform
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent

HASHED_GLOBS = [
    "medidor.py",
    "run_all.py",
    "tests/*.py",
    "configs/*.json",
    "data/*.json",
    "output/results.json",
]
"""What the claim depends on. Docs, lockfiles and CI config are informational only."""


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's bytes.

    Parameters
    ----------
    path : Path
        File to hash.

    Returns
    -------
    str
        64-character hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest() -> dict:
    """Hash every covered file and derive the single chain hash.

    Returns
    -------
    dict
        ``{"files": {relpath: sha256}, "chain_hash": str}`` with paths sorted so the
        chain is order-independent and reproducible on any machine.
    """
    files: dict[str, str] = {}
    for pattern in HASHED_GLOBS:
        for path in sorted(ROOT.glob(pattern)):
            files[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    concat = "".join(f"{name}:{digest}\n" for name, digest in sorted(files.items()))
    return {"files": files, "chain_hash": hashlib.sha256(concat.encode()).hexdigest()}


def write_outputs(manifest: dict) -> None:
    """Write ``provenance.json`` and the human-readable ``hash-chain.md``.

    Parameters
    ----------
    manifest : dict
        Output of :func:`build_manifest`.
    """
    out = ROOT / "output"
    out.mkdir(exist_ok=True)
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Provenance hash chain",
        "",
        f"**chain_hash:** `{manifest['chain_hash']}`",
        "",
        "Recompute and compare with `python make_provenance.py --verify`.",
        "",
        "## Hashed — the published numbers depend on these bytes",
        "",
        "| File | SHA-256 |",
        "|---|---|",
    ]
    lines += [f"| `{name}` | `{digest}` |" for name, digest in sorted(manifest["files"].items())]
    lines += [
        "",
        "## Informational — NOT hashed",
        "",
        f"- Generated: {date.today().isoformat()}",
        f"- Python: {platform.python_version()} on {platform.system()} {platform.machine()}",
        "- Documentation, `requirements.lock` and CI configuration are excluded so the",
        "  chain survives machine and toolchain changes. What is sealed is the program,",
        "  the frozen model configs, the published numbers and the derived results.",
    ]
    (out / "hash-chain.md").write_text("\n".join(lines) + "\n")


def verify() -> int:
    """Recompute the chain and compare it against the stored provenance.

    Returns
    -------
    int
        0 when the stored and recomputed chain hashes match, 1 otherwise.
    """
    stored_path = ROOT / "output" / "provenance.json"
    if not stored_path.exists():
        print("FAIL: output/provenance.json not found — run `python make_provenance.py` first")
        return 1
    stored = json.loads(stored_path.read_text())
    current = build_manifest()
    if stored["chain_hash"] == current["chain_hash"]:
        print(f"OK: chain_hash verified ({current['chain_hash'][:16]}...)")
        return 0
    print("FAIL: chain_hash mismatch")
    for name in sorted(set(stored["files"]) | set(current["files"])):
        if stored["files"].get(name) != current["files"].get(name):
            print(f"  changed: {name}")
    return 1


if __name__ == "__main__":
    if "--verify" in sys.argv:
        raise SystemExit(verify())
    manifest = build_manifest()
    write_outputs(manifest)
    print(f"chain_hash: {manifest['chain_hash']}")

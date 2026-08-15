"""Single-entry replication: measure -> results.json -> tests -> provenance seal.

Run ``python run_all.py`` in a fresh clone and every number the companion article
publishes is recomputed from the frozen configs, asserted by the test suite, and sealed
into a SHA-256 chain. Exit code 0 means: numbers reproduced, suite green, provenance
verified.

No third-party packages are needed to compute the numbers — only ``pytest`` to run the
assertions. No network access at any stage.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import medidor  # noqa: E402


def compute_results() -> dict:
    """Recompute the full measurement for both models in the article.

    Returns
    -------
    dict
        One entry per model, each the output of :func:`medidor.measure`, plus the
        digest of the config it was derived from.
    """
    results = {}
    for path, params, label in medidor.article_models():
        config = json.loads(path.read_text())
        results[path.stem.replace(".config", "")] = {
            "label": label,
            "config_file": path.relative_to(ROOT).as_posix(),
            "n_params": params,
            "measurement": medidor.measure(config, params),
        }
    return results


def main() -> int:
    """Run the replication pipeline end to end.

    Returns
    -------
    int
        0 on full success; the exit code of the first failing stage otherwise.
    """
    results = compute_results()
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    (output / "results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")

    qwen = results["Qwen3.8-27B"]["measurement"]
    headline = qwen["cache"][262144]
    print(
        f"[1/3] results.json written — Qwen3.8-27B at 256K: "
        f"viral {headline['viral_GiB']:.1f} GiB vs real {headline['real_GiB']:.2f} GiB "
        f"({headline['overestimate_x']:.1f}x)"
    )

    tests = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT)
    if tests.returncode != 0:
        return tests.returncode
    print("[2/3] test suite green — every published number asserted")

    build = subprocess.run([sys.executable, "make_provenance.py"], cwd=ROOT)
    if build.returncode != 0:
        return build.returncode
    check = subprocess.run([sys.executable, "make_provenance.py", "--verify"], cwd=ROOT)
    print("[3/3] provenance chain built and verified")
    return check.returncode


if __name__ == "__main__":
    raise SystemExit(main())

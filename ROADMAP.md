# Roadmap

The program does one thing and is finished for that purpose. What follows are honest
candidates, not commitments — each is listed with the condition that would justify it.

## Under consideration

**Multi-head Latent Attention (MLA).** DeepSeek-family models compress the cache into a
latent vector, breaking the width assumption in yet another way. Adding it requires a
frozen config to measure and a published model to verify against. *Trigger: a widely used
MLA model whose config declares the compression dimensions.*

**Per-layer report.** Currently the output aggregates by layer type. A per-layer table
would help someone auditing an unusual hybrid. *Trigger: a model whose layers of the same
type differ in width.*

**Activation and overhead estimates.** The one number in the article this program does
not derive is engine overhead. Deriving it properly means measuring real engines, which
is a different project with a different claim — it would need its own preregistration and
its own repository. *Trigger: not planned here; noted so the gap stays visible.*

**Replication notebook.** A Colab that clones, runs `run_all.py` and verifies the hash
chain with zero local setup. *Trigger: reader requests; the two-minute Track 1 path
already needs no install.*

## Explicitly not planned

- A web calculator. The article's argument is that opaque calculators are the problem.
- Support for formats that require loading model weights. This program reads
  configuration only, and that boundary is what keeps it dependency-free and fast.
- Vendoring more configs than the article measures. Every frozen file is a
  redistribution decision with its own license question; the two here are the two the
  article uses.

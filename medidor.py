#!/usr/bin/env python3
"""Medidor de memória de LLM local a partir do config.json REAL do modelo.

Motivo de existir: a fórmula que circula em cards virais
(`KV = batch x context x layers x hidden_size x 2 x bytes`) só vale para
Multi-Head Attention pura. Modelos abertos de 2026 usam GQA, atenção com janela
deslizante, atenção linear e K=V compartilhado — cada um deles derruba a conta.
Este script lê o config do modelo e calcula os dois números lado a lado.

Uso:
    python3 medidor.py fontes/config-Qwen3.8-27B.json --params 27781427952
    python3 medidor.py --repo Qwen/Qwen3.8-27B        # baixa config + params do HF

Sem argumento nenhum, roda os dois modelos do artigo a partir dos configs salvos.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import urllib.request
from pathlib import Path

GiB = 2**30
GB = 1e9
DOSSIE = Path(__file__).parent

# Cache que cresce com o contexto: só estes tipos de camada guardam um token por token.
# 'sliding_attention' cresce até o teto da janela e para; 'linear_attention' não cresce.
CRESCE_SEM_LIMITE = {"full_attention"}
CRESCE_ATE_A_JANELA = {"sliding_attention"}


def baixar(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def carregar(repo: str | None, caminho: str | None) -> tuple[dict, int | None, str]:
    """Devolve (config, n_params, rotulo). n_params vem dos safetensors quando disponível."""
    if repo:
        cfg = baixar(f"https://huggingface.co/{repo}/raw/main/config.json")
        try:
            meta = baixar(f"https://huggingface.co/api/models/{repo}")
            n = (meta.get("safetensors") or {}).get("total")
        except Exception:
            n = None
        return cfg, n, repo
    cfg = json.loads(Path(caminho).read_text())
    return cfg, None, Path(caminho).stem


def ficha(cfg: dict) -> dict:
    """Achata o config (Gemma/Qwen aninham tudo em text_config) e extrai o que importa."""
    t = cfg.get("text_config", cfg)
    tipos = t.get("layer_types") or []
    n_layers = t.get("num_hidden_layers") or len(tipos)
    if not tipos:  # modelo sem layer_types: todas as camadas são atenção cheia
        tipos = ["full_attention"] * n_layers

    n_kv = t.get("num_key_value_heads") or t.get("num_attention_heads")
    head_dim = t.get("head_dim")
    if not head_dim:  # fallback clássico quando o config não declara
        head_dim = t["hidden_size"] // t["num_attention_heads"]

    return {
        "layers": n_layers,
        "tipos": collections.Counter(tipos),
        "hidden": t["hidden_size"],
        "n_heads": t.get("num_attention_heads"),
        "n_kv": n_kv,
        "head_dim": head_dim,
        "global_head_dim": t.get("global_head_dim"),
        "janela": t.get("sliding_window"),
        # K=V compartilhado (Gemma 4): o cache guarda um tensor, não dois.
        "k_eq_v": bool(t.get("attention_k_eq_v")),
        "ctx_max": t.get("max_position_embeddings"),
        "linear": {
            "n_value_heads": t.get("linear_num_value_heads"),
            "k_dim": t.get("linear_key_head_dim"),
            "v_dim": t.get("linear_value_head_dim"),
            "dtype_bytes": 4 if t.get("mamba_ssm_dtype") == "float32" else 2,
        },
    }


def kv_por_token_por_camada(f: dict, bytes_cache: int, global_layer: bool = False) -> int:
    """Bytes de cache que UM token acrescenta em UMA camada de atenção com KV."""
    hd = f["global_head_dim"] if (global_layer and f["global_head_dim"]) else f["head_dim"]
    fator_kv = 1 if f["k_eq_v"] else 2  # K e V, ou um tensor só quando são iguais
    return fator_kv * f["n_kv"] * hd * bytes_cache


def estado_linear(f: dict) -> int:
    """Estado recorrente de UMA camada linear. Constante: não depende do contexto."""
    L = f["linear"]
    if not L["n_value_heads"]:
        return 0
    return L["n_value_heads"] * L["k_dim"] * L["v_dim"] * L["dtype_bytes"]


def cache_real(f: dict, ctx: int, batch: int, bytes_cache: int) -> int:
    total = 0
    for tipo, n in f["tipos"].items():
        if tipo in CRESCE_SEM_LIMITE:
            total += n * batch * ctx * kv_por_token_por_camada(f, bytes_cache, global_layer=True)
        elif tipo in CRESCE_ATE_A_JANELA:
            efetivo = min(ctx, f["janela"] or ctx)  # a janela é o teto do que se guarda
            total += n * batch * efetivo * kv_por_token_por_camada(f, bytes_cache)
        else:  # linear_attention e afins: estado fixo, sem batch de contexto
            total += n * batch * estado_linear(f)
    return total


def cache_do_card(f: dict, ctx: int, batch: int, bytes_cache: int) -> int:
    """A fórmula literal do card viral: batch x ctx x layers x hidden x 2 x bytes."""
    return batch * ctx * f["layers"] * f["hidden"] * 2 * bytes_cache


def relatorio(cfg: dict, n_params: int | None, rotulo: str, batch: int, bytes_cache: int) -> None:
    f = ficha(cfg)
    print(f"\n{'=' * 68}\n{rotulo}\n{'=' * 68}")
    print(f"camadas: {f['layers']} -> {dict(f['tipos'])}")
    print(f"hidden {f['hidden']} | heads {f['n_heads']} / kv {f['n_kv']} "
          f"(GQA {f['n_heads'] // f['n_kv']}x) | head_dim {f['head_dim']}"
          + (f" (global {f['global_head_dim']})" if f["global_head_dim"] else "")
          + (" | K=V compartilhado" if f["k_eq_v"] else "")
          + (f" | janela {f['janela']}" if f["janela"] else ""))

    if n_params:
        print(f"\nPESOS ({n_params:,} params reais):")
        for nome, bpp in [("BF16", 2), ("INT8", 1), ("4-bit ideal", 0.5), ("Q4_K_M (4,89 bpw)", 4.89 / 8)]:
            b = n_params * bpp
            print(f"  {nome:20} {b / GB:7.2f} GB | {b / GiB:7.2f} GiB")

    est = estado_linear(f)
    if est:
        n_lin = sum(n for t, n in f["tipos"].items() if t not in CRESCE_SEM_LIMITE | CRESCE_ATE_A_JANELA)
        print(f"\nestado linear: {est / 2**20:.1f} MiB/camada x {n_lin} = "
              f"{n_lin * est / GiB:.2f} GiB CONSTANTE (não cresce com o contexto)")

    print(f"\ncache com batch={batch}, {bytes_cache} byte(s)/elemento:")
    print(f"{'contexto':>10} | {'fórmula do card':>16} | {'real':>12} | {'erro':>6}")
    print("-" * 54)
    for ctx in (1024, 8192, 32768, 131072, 262144):
        if f["ctx_max"] and ctx > f["ctx_max"]:
            continue
        card, real = cache_do_card(f, ctx, batch, bytes_cache), cache_real(f, ctx, batch, bytes_cache)
        print(f"{ctx // 1024:>9}K | {card / GiB:12.2f} GiB | {real / GiB:8.2f} GiB | {card / real:5.1f}x")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("config", nargs="?", help="caminho de um config.json salvo")
    p.add_argument("--repo", help="repo do HuggingFace (baixa config + contagem de params)")
    p.add_argument("--params", type=int, help="parâmetros totais, quando o config vem de arquivo")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--bytes-cache", type=int, default=2, help="2=BF16, 1=cache quantizado em 8 bits")
    a = p.parse_args()

    if not a.config and not a.repo:  # sem argumento: os dois modelos do artigo
        padrao = [
            (DOSSIE / "fontes/config-Qwen3.8-27B.json", 27_781_427_952, "Qwen3.8-27B (exemplo do artigo)"),
            (DOSSIE / "fontes/config-google_gemma-4-26B-A4B-it.json", None, "Gemma 4 26B A4B (exemplo do card)"),
        ]
        achou = False
        for caminho, params, rotulo in padrao:
            if caminho.exists():
                relatorio(json.loads(caminho.read_text()), params, rotulo, a.batch, a.bytes_cache)
                achou = True
            else:
                print(f"[faltando] {caminho}", file=sys.stderr)
        return 0 if achou else 1

    cfg, n, rotulo = carregar(a.repo, a.config)
    relatorio(cfg, a.params or n, rotulo, a.batch, a.bytes_cache)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Testa se as macro-regiões são um resultado estável ou um acidente da execução.

O Louvain é estocástico e o projeto fixa `seed=42`. Isso levanta duas perguntas que a banca
pode fazer, e as duas têm resposta numérica:

1. **A semente foi escolhida a dedo?** Roda com N sementes e compara cada partição com a do
   projeto pelo ARI (Adjusted Rand Index): 1,0 = idênticas, 0,0 = tão parecidas quanto o acaso.
2. **A modularidade 0,40 é alta?** Q não é comparável entre redes de densidades diferentes.
   Numa rede densa como esta, o acaso já produz Q alto (Guimerà et al., 2004), então o número
   sozinho não diz nada — o que diz é a distância até o modelo nulo.

Também compara as partições obtidas com cada peso de aresta disponível (`q_calls`, `J`,
`n_pairs`), porque a escolha do peso mexe mais no resultado do que a semente.

Exemplos:
    python scripts/estabilidade_macroregioes.py campinas
    python scripts/estabilidade_macroregioes.py campinas --seeds 100 --nulos 50
    python scripts/estabilidade_macroregioes.py campinas --pular-nulos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import networkx as nx  # noqa: E402
from networkx.algorithms.community import louvain_communities, modularity  # noqa: E402

from config import load_config  # noqa: E402
from src import antenna as antenna_module  # noqa: E402

PESOS = ["q_calls", "J", "n_pairs"]


def _comb2(x):
    return x * (x - 1) / 2


def ari(a, b) -> float:
    """Adjusted Rand Index entre duas partições (Hubert & Arabie, 1985).

    Corrige o Rand Index pelo acaso: partições independentes dão ≈ 0, idênticas dão 1. Ignora
    o nome dos rótulos — só importa quem ficou junto com quem.
    """
    ct = pd.crosstab(pd.Series(a), pd.Series(b)).to_numpy()
    n = ct.sum()
    sij, si, sj = _comb2(ct).sum(), _comb2(ct.sum(1)).sum(), _comb2(ct.sum(0)).sum()
    esperado = si * sj / _comb2(n)
    maximo = (si + sj) / 2
    return float((sij - esperado) / (maximo - esperado)) if maximo != esperado else 1.0


def rotular(comunidades, ordem) -> list[int]:
    de_quem = {n: i for i, c in enumerate(comunidades) for n in c}
    return [de_quem[n] for n in ordem]


def construir(cidade: str, peso: str | None):
    config = load_config(cidade)
    if peso:
        config.setdefault("antenna", {})["weight"] = peso
    data = config["data"]
    ea, an = ROOT / data["edges_antenna_path"], ROOT / data["antennas_path"]
    if not (ea.exists() and an.exists()):
        raise FileNotFoundError(
            f"parquets por antena não encontrados ({ea.name}, {an.name}). "
            f"Rode antes: python main.py --city {cidade} --analyses eda"
        )
    net = antenna_module.build(pd.read_parquet(ea), pd.read_parquet(an), config)
    return net, config.get("antenna", {}).get("weight", "q_calls")


def testar_sementes(G, seed_base: int, n: int) -> dict:
    """Roda o Louvain com n sementes e compara tudo contra a partição do projeto."""
    ordem = sorted(G.nodes())
    base = louvain_communities(G, weight="weight", seed=seed_base)
    ref = rotular(base, ordem)

    aris, ns, qs = [], [], []
    for s in range(n):
        c = louvain_communities(G, weight="weight", seed=s)
        aris.append(ari(ref, rotular(c, ordem)))
        ns.append(len(c))
        qs.append(modularity(G, c, weight="weight"))

    return {
        "q_base": modularity(G, base, weight="weight"),
        "n_base": len(base),
        "aris": np.array(aris), "ns": np.array(ns), "qs": np.array(qs),
    }


def testar_nulos(G, n: int, rng) -> dict:
    """Q que o acaso produz nesta densidade — a régua que falta para ler a modularidade."""
    pesos = np.array([d["weight"] for _, _, d in G.edges(data=True)])

    embaralhado = []
    for i in range(n):
        H = G.copy()
        for (u, v), w in zip(H.edges(), rng.permutation(pesos)):
            H[u][v]["weight"] = float(w)
        embaralhado.append(modularity(H, louvain_communities(H, weight="weight", seed=i),
                                      weight="weight"))

    aleatorio = []
    for i in range(n):
        R = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges(), seed=i)
        for (u, v), w in zip(R.edges(), rng.permutation(pesos)):
            R[u][v]["weight"] = float(w)
        aleatorio.append(modularity(R, louvain_communities(R, weight="weight", seed=i),
                                    weight="weight"))

    return {"embaralhado": np.array(embaralhado), "aleatorio": np.array(aleatorio)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cidade", help="identificador da cidade (config/<cidade>.yaml)")
    p.add_argument("--seeds", type=int, default=30, help="quantas sementes testar")
    p.add_argument("--nulos", type=int, default=30, help="repetições de cada modelo nulo")
    p.add_argument("--seed-base", type=int, default=42, help="a semente usada pelo projeto")
    p.add_argument("--pular-nulos", action="store_true", help="só o teste de sementes (mais rápido)")
    p.add_argument("--pular-pesos", action="store_true", help="não comparar os pesos de aresta")
    args = p.parse_args()

    try:
        net, peso = construir(args.cidade, None)
    except FileNotFoundError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    G = net.G

    print(f"Rede de regiões: {G.number_of_nodes()} nós, {G.number_of_edges()} fluxos, "
          f"densidade {nx.density(G):.3f}, peso={peso}\n")

    # ---------------------------------------------------------------- sementes
    r = testar_sementes(G, args.seed_base, args.seeds)
    print(f"── A semente importa? ({args.seeds} sementes) " + "─" * 30)
    print(f"partição do projeto (seed={args.seed_base}): {r['n_base']} macro-regiões, "
          f"Q={r['q_base']:.3f}")
    unicos = sorted(set(r["ns"].tolist()))
    print(f"nº de macro-regiões: {'sempre ' + str(unicos[0]) if len(unicos) == 1 else unicos}")
    print(f"Q: {r['qs'].min():.3f}–{r['qs'].max():.3f}")
    print(f"ARI contra a partição do projeto: mediano {np.median(r['aris']):.3f}, "
          f"mínimo {r['aris'].min():.3f}")
    if r["aris"].min() > 0.9:
        print("→ partição estável: a semente não muda a conclusão.")
    elif r["aris"].min() > 0.6:
        print("→ estabilidade moderada: o núcleo é estável, as fronteiras oscilam.")
    else:
        print("→ ATENÇÃO: a partição depende da semente. Não apresente como resultado único.")

    # ---------------------------------------------------------------- nulos
    if not args.pular_nulos:
        rng = np.random.default_rng(args.seed_base)
        nul = testar_nulos(G, args.nulos, rng)
        emb, ale = nul["embaralhado"], nul["aleatorio"]
        piso = max(emb.mean(), ale.mean())
        desvio = emb.std() if emb.mean() >= ale.mean() else ale.std()
        z = (r["q_base"] - piso) / desvio if desvio > 0 else float("inf")
        print(f"\n── Q={r['q_base']:.3f} é alto? ({args.nulos} repetições) " + "─" * 24)
        print(f"pesos embaralhados (topologia intacta): Q = {emb.mean():.3f} ± {emb.std():.3f}")
        print(f"grafo aleatório de mesma densidade:     Q = {ale.mean():.3f} ± {ale.std():.3f}")
        print(f"observado / nulo: {r['q_base'] / piso:.2f}×  |  z = {z:.1f} desvios-padrão")
        print("→ numa rede densa o acaso já produz Q alto; o que vale é a distância até o nulo,")
        print("  não o valor absoluto de Q.")

    # ---------------------------------------------------------------- pesos
    if not args.pular_pesos:
        print(f"\n── O peso da aresta importa? " + "─" * 37)
        parts, info = {}, {}
        for w in PESOS:
            try:
                n2, _ = construir(args.cidade, w)
            except Exception as exc:                      # peso ausente nos fluxos
                print(f"  {w}: indisponível ({exc})")
                continue
            ordem = sorted(n2.G.nodes())
            c = louvain_communities(n2.G, weight="weight", seed=args.seed_base)
            parts[w] = rotular(c, ordem)
            info[w] = (len(c), modularity(n2.G, c, weight="weight"))
        for w, (n_com, q) in info.items():
            print(f"  peso={w:8s} → {n_com} macro-regiões, Q={q:.3f}")
        print()
        nomes = list(parts)
        for i in range(len(nomes)):
            for j in range(i + 1, len(nomes)):
                print(f"  ARI {nomes[i]} vs {nomes[j]}: {ari(parts[nomes[i]], parts[nomes[j]]):.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

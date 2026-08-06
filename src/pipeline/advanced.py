"""Análises estruturais da rede de regiões: robustez, rich-club e efeito da agregação.

Substitui o módulo antigo (lei de potência, small-world contra Erdős–Rényi, k-core,
fragmentação da componente gigante). Nenhuma dessas métricas sobrevive à mudança de unidade:
com 145 nós e densidade 0,56 não há cauda de grau para ajustar, o caminho médio já é ~1,4
e a rede simplesmente não fragmenta. O que substitui cada uma:

- fragmentação  →  perda de **eficiência ponderada** ao remover regiões
- k-core        →  s-core (em `topology`) e **rich-club por força** aqui
- lei de potência →  desigualdade de volume (Gini/Lorenz, em `topology`)

A última seção compara o nível individual com o regional — é a defesa contra a falácia
ecológica e, no caso de Campinas, muda a leitura do principal achado do projeto.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from ..antenna import weighted_global_efficiency
from ..graph_builder import build_edges_graph, build_user_antenna_map

logger = logging.getLogger("pipeline")


# --------------------------------------------------------------------------- robustez
def _efficiency_curve(G: nx.Graph, order: list, max_frac: float = 0.6, steps: int = 20):
    """Eficiência ponderada (relativa à rede intacta) conforme regiões são removidas."""
    n = G.number_of_nodes()
    base = weighted_global_efficiency(G)
    fracs = np.linspace(0, max_frac, steps)
    out = []
    for f in fracs:
        k = int(f * n)
        H = G.subgraph([node for node in G.nodes() if node not in set(order[:k])])
        out.append(weighted_global_efficiency(H) / base if base > 0 else np.nan)
    return fracs, np.array(out)


def _robustness(G: nx.Graph, config: dict, exporter, city_name: str) -> dict:
    """Ataque dirigido às regiões de maior volume vs. perda aleatória de regiões."""
    if G.number_of_nodes() < 5 or G.number_of_edges() == 0:
        logger.warning("[advanced] rede pequena demais — robustez pulada")
        return {}

    steps = int(config.get("advanced", {}).get("robustness_steps", 20))
    strength_order = [u for u, _ in sorted(G.degree(weight="weight"), key=lambda kv: -kv[1])]
    random_order = list(G.nodes())
    random.Random(1).shuffle(random_order)

    fracs, attack = _efficiency_curve(G, strength_order, steps=steps)
    _, failure = _efficiency_curve(G, random_order, steps=steps)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(fracs, attack, "o-", color="crimson", label="perda das regiões de maior volume")
    ax.plot(fracs, failure, "s-", color="steelblue", label="perda aleatória de regiões")
    ax.axhline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("fração das regiões removidas")
    ax.set_ylabel("eficiência de comunicação (relativa à rede intacta)")
    ax.set_title(f"Robustez da rede de regiões — {city_name}")
    ax.legend()
    exporter.save_figure(fig, "robustness", "advanced")

    def _collapse(curve):
        hit = [f for f, y in zip(fracs, curve) if y < 0.5]
        return float(hit[0]) if hit else None

    return {
        "efficiency_attack_collapse": _collapse(attack),
        "efficiency_failure_collapse": _collapse(failure),
        "efficiency_gap_at_20pct": float(
            np.interp(0.2, fracs, failure) - np.interp(0.2, fracs, attack)
        ),
    }


# --------------------------------------------------------------------------- rich-club
def _weighted_rich_club(G: nx.Graph, exporter, city_name: str, n_reps: int = 50) -> dict:
    """Rich-club ponderado (Opsahl et al., 2008): as regiões mais ativas falam entre si?

    φ^w(r) = peso trocado entre os nós de força acima de *r*, dividido pelo peso que essas
    mesmas ligações teriam se levassem as arestas mais pesadas da rede. O nulo embaralha os
    pesos sobre a mesma topologia — apropriado aqui, já que a topologia é quase completa e
    toda a informação está nos pesos.
    """
    if G.number_of_edges() < 20:
        return {}

    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)], dtype=float)
    sorted_w = np.sort(weights)[::-1]
    strength = dict(G.degree(weight="weight"))

    def phi_curve(edge_weights: dict) -> tuple[np.ndarray, np.ndarray]:
        s = {node: 0.0 for node in G.nodes()}
        for (u, v), w in edge_weights.items():
            s[u] += w
            s[v] += w
        ranks = np.unique(np.quantile(list(s.values()), np.linspace(0.1, 0.95, 25)))
        phis = []
        for r in ranks:
            club = {node for node, value in s.items() if value > r}
            inside = [w for (u, v), w in edge_weights.items() if u in club and v in club]
            if len(inside) < 3:
                phis.append(np.nan)
                continue
            top = np.sort(np.fromiter(edge_weights.values(), dtype=float))[::-1][: len(inside)]
            phis.append(float(np.sum(inside) / np.sum(top)) if top.sum() else np.nan)
        return ranks, np.array(phis)

    observed = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    ranks, phi_real = phi_curve(observed)

    rng = np.random.default_rng(42)
    keys = list(observed)
    null_curves = []
    for _ in range(n_reps):
        shuffled = dict(zip(keys, rng.permutation(list(observed.values()))))
        null_curves.append(phi_curve(shuffled)[1])
    stacked = np.vstack(null_curves)
    # em cidades pequenas alguns limiares não têm clube nenhum: a coluna é toda NaN
    phi_null = np.full(stacked.shape[1], np.nan)
    usable = ~np.all(np.isnan(stacked), axis=0)
    phi_null[usable] = np.nanmean(stacked[:, usable], axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        rho = phi_real / phi_null

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].plot(ranks, phi_real, "o-", color="crimson", label="rede real")
    axes[0].plot(ranks, phi_null, "s-", color="steelblue", label="pesos embaralhados")
    axes[0].set_xlabel("limiar de força  r (chamadas)")
    axes[0].set_ylabel("φ$^w$(r)")
    axes[0].set_title("Rich-club ponderado")
    axes[0].legend()
    axes[1].axhline(1.0, color="gray", ls="--", lw=1)
    axes[1].plot(ranks, rho, "o-", color="purple")
    axes[1].set_xlabel("limiar de força  r")
    axes[1].set_ylabel("ρ(r) = φ real / φ nulo")
    axes[1].set_title("ρ > 1 indica clube das regiões mais ativas")
    fig.tight_layout()
    exporter.save_figure(fig, "rich_club", "advanced")

    top_rho = rho[~np.isnan(rho)][-max(1, len(rho) // 4):]
    return {"rich_club_ratio_high_strength": float(np.mean(top_rho))} if len(top_rho) else {}


# --------------------------------------------------------------------------- agregação
def _individual_vs_regional(edges_antenna: pd.DataFrame, net, exporter, city_name: str) -> dict:
    """Compara a homofilia socioeconômica no nível da pessoa e no nível da região.

    O achado antigo — "49% das chamadas ligam pessoas do mesmo quintil, contra 26% ao acaso"
    — usava um nulo que embaralha o quintil **entre pessoas**, destruindo junto o fato de que
    vizinhos compartilham o quintil por morarem no mesmo lugar. Repetindo a conta com um nulo
    que embaralha o quintil **entre regiões** (preservando quem mora com quem), separa-se o
    que é preferência social do que é simples proximidade territorial.
    """
    quintile_by_antenna = net.nodes.set_index("antenna_id")["residence_quintile_state"]
    user_antenna = build_user_antenna_map(edges_antenna)
    user_quintile = user_antenna.map(quintile_by_antenna)

    pairs = build_edges_graph(edges_antenna)
    pairs["q_source"] = pairs["source"].map(user_quintile)
    pairs["q_target"] = pairs["target"].map(user_quintile)
    pairs = pairs.dropna(subset=["q_source", "q_target"])
    if pairs.empty or pairs["q_source"].nunique() < 2:
        return {}

    w = pairs["q_calls"].to_numpy(dtype=float)
    total = w.sum()
    observed = float(w[(pairs["q_source"] == pairs["q_target"]).to_numpy()].sum() / total)

    rng = np.random.default_rng(42)

    # nulo 1 — embaralha o quintil entre pessoas (o do pipeline antigo)
    users = pd.Index(user_quintile.index)
    codes = user_quintile.to_numpy()
    si, ti = users.get_indexer(pairs["source"]), users.get_indexer(pairs["target"])
    valid = (si >= 0) & (ti >= 0)
    null_user = float(np.mean([
        w[valid][perm[si[valid]] == perm[ti[valid]]].sum() / total
        for perm in (rng.permutation(codes) for _ in range(50))
    ]))

    # nulo 2 — embaralha o quintil entre regiões, preservando a composição de cada antena
    antennas = pd.Index(net.nodes["antenna_id"])
    ant_codes = quintile_by_antenna.reindex(net.nodes["antenna_id"]).to_numpy()
    ai = antennas.get_indexer(pairs["source"].map(user_antenna))
    bi = antennas.get_indexer(pairs["target"].map(user_antenna))
    ok = (ai >= 0) & (bi >= 0)
    null_spatial = float(np.mean([
        w[ok][perm[ai[ok]] == perm[bi[ok]]].sum() / total
        for perm in (rng.permutation(ant_codes) for _ in range(50))
    ]))

    ratio_user = observed / null_user if null_user else float("nan")
    ratio_spatial = observed / null_spatial if null_spatial else float("nan")

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["observado", "acaso\n(quintil entre pessoas)", "acaso\n(quintil entre regiões)"]
    values = [observed, null_user, null_spatial]
    bars = ax.bar(labels, values, color=["crimson", "lightgray", "steelblue"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.0%}",
                ha="center", fontsize=11)
    ax.set_ylabel("fração do volume entre pessoas do mesmo quintil")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title(f"Homofilia: preferência social ou proximidade territorial? — {city_name}")
    exporter.save_figure(fig, "homophily_levels", "advanced")

    return {
        "individual_homophily_observed": observed,
        "individual_homophily_null_by_user": null_user,
        "individual_homophily_null_by_region": null_spatial,
        "individual_homophily_ratio_naive": float(ratio_user),
        "individual_homophily_ratio_spatial_null": float(ratio_spatial),
    }


# --------------------------------------------------------------------------- run
def run(net, edges_antenna: pd.DataFrame, config: dict, exporter) -> dict:
    """Robustez ponderada, rich-club por força e comparação entre níveis de agregação."""
    logger.info("[advanced] robustez ponderada, rich-club, individual vs regional")
    city_name = exporter.city_name
    metrics: dict = {}

    metrics.update(_robustness(net.G, config, exporter, city_name))
    metrics.update(_weighted_rich_club(net.G, exporter, city_name))
    metrics.update(_individual_vs_regional(edges_antenna, net, exporter, city_name))
    exporter.add_metrics("advanced", metrics)

    partes = []
    if "efficiency_attack_collapse" in metrics:
        collapse = metrics["efficiency_attack_collapse"]
        gap = metrics.get("efficiency_gap_at_20pct")
        if collapse is not None:
            partes.append(
                f"A rede de regiões **não fragmenta** — ela perde capacidade aos poucos. Removendo "
                f"as regiões de maior volume, a eficiência de comunicação cai à metade com "
                f"**{collapse:.0%}** das regiões fora; a perda aleatória é bem mais benigna "
                f"(diferença de {gap:.0%} em favor do acaso com 20% removidas)."
            )
        else:
            partes.append(
                "A rede de regiões degrada suavemente: mesmo removendo as áreas de maior volume, "
                "a eficiência de comunicação se mantém acima da metade em toda a faixa testada."
            )
    if "rich_club_ratio_high_strength" in metrics:
        rc = metrics["rich_club_ratio_high_strength"]
        verbo = "formam" if rc > 1.05 else "não formam"
        partes.append(
            f"As regiões de maior tráfego **{verbo} um rich-club** (ρ ≈ {rc:.2f} no topo da "
            f"distribuição de força)."
        )
    if "individual_homophily_ratio_naive" in metrics:
        partes.append(
            f"**Sobre a homofilia socioeconômica:** no nível das pessoas, "
            f"{metrics['individual_homophily_observed']:.0%} do volume liga o mesmo quintil, contra "
            f"{metrics['individual_homophily_null_by_user']:.0%} de um acaso que embaralha o quintil "
            f"entre indivíduos (**{metrics['individual_homophily_ratio_naive']:.1f}×**). Mas esse nulo "
            f"também destrói o fato de que vizinhos compartilham quintil por morarem no mesmo lugar. "
            f"Com um acaso que embaralha o quintil **entre regiões**, o esperado sobe para "
            f"{metrics['individual_homophily_null_by_region']:.0%} e a razão cai para "
            f"**{metrics['individual_homophily_ratio_spatial_null']:.2f}×** — ou seja, a maior parte da "
            f"“segregação socioeconômica na comunicação” é, na verdade, **segregação territorial**: "
            f"as pessoas falam com quem está perto, e quem está perto tem a mesma renda."
        )
    if partes:
        exporter.add_report_section("Robustez, rich-club e efeito da agregação", " ".join(partes))

    return {"metrics": metrics}

"""Análise de topologia da rede (lógica do notebook 2-rede-complexa)."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import louvain_communities, modularity

logger = logging.getLogger("pipeline")


def run(G: nx.Graph, G_main: nx.Graph, config: dict, exporter) -> dict:
    """Calcula e exporta as métricas de topologia da rede."""
    logger.info("[topology] grau, componentes, clustering, centralidades, comunidades")
    k_sample = config.get("advanced", {}).get("betweenness_sample_k", 500)

    # -------- métricas de grafo (rede completa) --------
    n, m = G.number_of_nodes(), G.number_of_edges()
    density = nx.density(G)
    component_sizes = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
    giant = G_main.number_of_nodes()
    exporter.add_metrics(
        "graph",
        {
            "nodes": n,
            "edges": m,
            "density": density,
            "n_components": len(component_sizes),
            "giant_nodes": giant,
            "giant_fraction_pct": round(100 * giant / n, 1),
        },
    )

    # -------- distribuição de grau --------
    degrees = np.array([d for _, d in G.degree()])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(degrees, bins=50, log=True)
    ax.set_xlabel("grau")
    ax.set_ylabel("frequência (escala log)")
    ax.set_title(f"Distribuição de grau — {exporter.city_name}")
    exporter.save_figure(fig, "degree_distribution", "topology")

    x = np.sort(np.unique(degrees))
    ccdf = np.array([np.mean(degrees >= k) for k in x])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(x, ccdf, "o", ms=4, alpha=0.7)
    ax.set_xlabel("grau  k")
    ax.set_ylabel("P(K ≥ k)")
    ax.set_title("CCDF da distribuição de grau")
    exporter.save_figure(fig, "ccdf", "topology")

    # -------- clustering --------
    avg_clustering = nx.average_clustering(G)

    # -------- centralidades (componente gigante) --------
    deg_c = dict(G_main.degree())
    strength = dict(G_main.degree(weight="q_calls"))
    betw = nx.betweenness_centrality(
        G_main, k=min(k_sample, G_main.number_of_nodes()), weight=None, seed=42
    )
    eig = nx.eigenvector_centrality(G_main, max_iter=1000, weight="weight")

    centralidades = pd.DataFrame({"user_id": list(G_main.nodes())})
    centralidades["grau"] = centralidades["user_id"].map(deg_c)
    centralidades["forca_chamadas"] = centralidades["user_id"].map(strength)
    centralidades["intermediacao"] = centralidades["user_id"].map(betw)
    centralidades["autovetor"] = centralidades["user_id"].map(eig)
    exporter.save_data(
        centralidades.sort_values("grau", ascending=False).head(20), "top_hubs.csv"
    )

    # -------- comunidades (Louvain) --------
    communities = louvain_communities(G_main, weight="weight", seed=42)
    Q = modularity(G_main, communities, weight="weight")
    community_sizes = sorted((len(c) for c in communities), reverse=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(community_sizes, bins=30)
    ax.set_xlabel("tamanho da comunidade")
    ax.set_ylabel("número de comunidades")
    ax.set_title(f"Comunidades (Louvain): {len(communities)} | modularidade {Q:.3f}")
    exporter.save_figure(fig, "communities", "topology")

    metrics = {
        "degree_mean": float(degrees.mean()),
        "degree_median": float(np.median(degrees)),
        "degree_max": int(degrees.max()),
        "avg_clustering": float(avg_clustering),
        "n_communities": len(communities),
        "largest_community": int(community_sizes[0]),
        "modularity": float(Q),
    }
    exporter.add_metrics("topology", metrics)

    exporter.add_report_section(
        "Topologia",
        f"A rede tem **{n:,} nós** e **{m:,} arestas** (densidade {density:.2e}); a componente "
        f"gigante reúne **{giant:,} nós ({100 * giant / n:.1f}%)**. A distribuição de grau é "
        f"concentrada (mediana {np.median(degrees):.0f}, máximo {degrees.max()}), com clustering "
        f"médio **{avg_clustering:.2f}** — bem acima do aleatório. O algoritmo de Louvain detecta "
        f"**{len(communities)} comunidades** (a maior com {community_sizes[0]} usuários) e "
        f"**modularidade {Q:.3f}**, indicando uma estrutura fortemente modular.",
    )

    return {"centralidades": centralidades, "metrics": metrics}

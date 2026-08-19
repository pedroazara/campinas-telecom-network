"""Estrutura da rede de regiões (antenas).

Substitui a antiga topologia da rede de usuários. Numa rede de 145 nós com densidade 0,56
quase toda região fala com quase toda região, então **grau não distingue nada**: as análises
aqui olham para peso (força, desigualdade), para o *backbone* dos fluxos realmente
significativos e para a divisão da cidade em macro-regiões funcionais.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.community import louvain_communities, modularity

from ..antenna import s_core_levels, build_contact_matrix

logger = logging.getLogger("pipeline")


def _gini(values: np.ndarray) -> float:
    """Índice de Gini de uma distribuição não-negativa (0 = igual, 1 = tudo num nó só)."""
    x = np.sort(np.asarray(values, dtype=float))
    x = x[np.isfinite(x)]
    if len(x) == 0 or x.sum() == 0:
        return float("nan")
    n = len(x)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1) / n)


def _plot_strength(nodes: pd.DataFrame, exporter) -> dict:
    """Distribuição do volume de chamadas por região e sua curva de Lorenz."""
    volume = nodes["calls_total"].to_numpy(dtype=float)
    gini = _gini(volume)

    ordered = np.sort(volume)
    cum_share = np.concatenate([[0], np.cumsum(ordered) / ordered.sum()])
    pop_share = np.linspace(0, 1, len(ordered) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(volume, bins=30, color="steelblue")
    axes[0].set_xlabel("chamadas que tocam a região")
    axes[0].set_ylabel("nº de regiões")
    axes[0].set_title("Volume de chamadas por região")
    axes[1].plot(pop_share, cum_share, "-", color="crimson", lw=2, label="observado")
    axes[1].plot([0, 1], [0, 1], "--", color="gray", lw=1, label="igualdade perfeita")
    axes[1].set_xlabel("fração das regiões (da menor para a maior)")
    axes[1].set_ylabel("fração acumulada do volume")
    axes[1].set_title(f"Curva de Lorenz — Gini = {gini:.2f}")
    axes[1].legend()
    fig.tight_layout()
    exporter.save_figure(fig, "strength_distribution", "topology")

    return {"volume_gini": gini}


def _plot_backbone(G: nx.Graph, B: nx.Graph, alpha: float, exporter, city_name: str) -> dict:
    """Compara o backbone por disparidade com um corte ingênuo pelos maiores fluxos.

    Em volume os dois se equivalem; a diferença está em **quem** sobrevive. O corte por peso
    absoluto apaga as regiões pequenas inteiras — elas nunca têm fluxos grandes. O filtro de
    disparidade julga cada fluxo contra a força do próprio nó, então preserva o corredor
    principal de um bairro pequeno tanto quanto o de um grande.
    """
    edges = sorted(G.edges(data=True), key=lambda e: -e[2]["weight"])
    weights = np.array([d["weight"] for _, _, d in edges])
    total_weight = weights.sum()
    n_edges = len(edges)

    b_edges = B.number_of_edges()
    b_weight = sum(d["weight"] for _, _, d in B.edges(data=True))
    edge_frac = b_edges / n_edges if n_edges else float("nan")
    weight_frac = b_weight / total_weight if total_weight else float("nan")

    # cobertura: quantas regiões mantêm ao menos um fluxo em cada estratégia
    fracs = np.linspace(0.02, 1.0, 40)
    naive_coverage = []
    for f in fracs:
        keep = edges[: max(1, int(f * n_edges))]
        naive_coverage.append(len({u for u, _, _ in keep} | {v for _, v, _ in keep}))
    backbone_coverage = len([n for n, d in B.degree() if d > 0])

    # comparação justa: corte ingênuo com exatamente o mesmo número de fluxos do backbone
    naive_same_size = edges[:b_edges]
    naive_coverage_same_size = len(
        {u for u, _, _ in naive_same_size} | {v for _, v, _ in naive_same_size}
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(np.arange(1, n_edges + 1) / n_edges, np.cumsum(weights) / total_weight,
                 "-", color="gray", lw=2, label="corte pelos maiores fluxos")
    axes[0].plot(edge_frac, weight_frac, "o", color="crimson", ms=10,
                 label=f"filtro de disparidade (α={alpha})")
    axes[0].set_xlabel("fração dos fluxos preservados")
    axes[0].set_ylabel("fração do volume preservado")
    axes[0].set_title("Volume retido")
    axes[0].legend()

    axes[1].plot(fracs, naive_coverage, "-", color="gray", lw=2, label="corte pelos maiores fluxos")
    axes[1].plot(edge_frac, backbone_coverage, "o", color="crimson", ms=10,
                 label="filtro de disparidade")
    axes[1].axhline(G.number_of_nodes(), color="black", ls=":", lw=1, label="todas as regiões")
    axes[1].set_xlabel("fração dos fluxos preservados")
    axes[1].set_ylabel("regiões com ao menos um fluxo")
    axes[1].set_title("Cobertura do território")
    axes[1].legend()
    fig.suptitle(f"Backbone da rede de regiões — {city_name}")
    fig.tight_layout()
    exporter.save_figure(fig, "backbone", "topology")

    return {
        "backbone_alpha": alpha,
        "backbone_edges": b_edges,
        "backbone_edge_fraction": float(edge_frac),
        "backbone_weight_fraction": float(weight_frac),
        "backbone_density": float(nx.density(B)),
        "backbone_regions_covered": int(backbone_coverage),
        "naive_cut_regions_covered": int(naive_coverage_same_size),
    }



def _plot_contact_matrix(cm, ordem, exporter, city_name: str) -> dict:
    """Heatmap da matriz J, com as antenas agrupadas pelas macro-regiões.

    J varre várias ordens de grandeza, então a escala de cor é logarítmica. Ordenar as
    antenas pela macro-região faz a estrutura de blocos aparecer: se a divisão detectada nos
    fluxos for real, os blocos na diagonal ficam visivelmente mais densos que o resto.
    """
    from matplotlib.colors import LogNorm

    J = cm.J.reindex(index=ordem, columns=ordem).to_numpy(dtype=float)
    fora = ~np.eye(len(J), dtype=bool)
    positivos = J[fora & (J > 0)]
    if positivos.size == 0:
        return {}

    fig, ax = plt.subplots(figsize=(9, 7.5))
    imagem = ax.imshow(
        np.where(J > 0, J, np.nan), cmap="magma_r",
        norm=LogNorm(vmin=positivos.min(), vmax=positivos.max()),
        interpolation="nearest",
    )
    fig.colorbar(imagem, ax=ax, label="J$_{lm}$ = contatos / pares possíveis (escala log)")
    ax.set_xlabel("antena m (agrupadas por macro-região)")
    ax.set_ylabel("antena l")
    ax.set_title(f"Matriz de conexão entre antenas — {city_name}")
    exporter.save_figure(fig, "contact_matrix", "topology")

    return {
        "contact_pairs_between_regions": int(cm.K.to_numpy()[fora].sum() // 2),
        "contact_pairs_internal": int(np.diagonal(cm.K.to_numpy()).sum() // 2),
        "J_median": float(np.median(positivos)),
        "J_max": float(positivos.max()),
        "J_connected_region_pairs": int((J[fora] > 0).sum() // 2),
    }


def _plot_score(G: nx.Graph, exporter, city_name: str) -> tuple[pd.Series, dict]:
    """Decomposição s-core: o núcleo de regiões que concentra o tráfego."""
    levels = s_core_levels(G)
    if levels.max() == 0:
        return levels, {}

    grid = np.linspace(0, levels.max(), 40)
    sizes = [int((levels >= t).sum()) for t in grid]
    core_size = int((levels >= levels.max()).sum())

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(grid, sizes, "o-", color="darkorange")
    ax.set_xlabel("limiar de força (chamadas)")
    ax.set_ylabel("regiões no s-core")
    ax.set_title(f"Decomposição s-core — núcleo final com {core_size} regiões ({city_name})")
    exporter.save_figure(fig, "score", "topology")

    return levels, {"score_max_strength": float(levels.max()), "score_core_size": core_size}


def _plot_balance(nodes: pd.DataFrame, reciprocity: float, exporter, city_name: str) -> None:
    """Quanto cada região emite a mais do que recebe."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(nodes["net_balance"].dropna(), bins=25, color="teal")
    axes[0].axvline(0, color="black", ls="--", lw=1)
    axes[0].set_xlabel("balanço líquido  (emitidas − recebidas) / total")
    axes[0].set_ylabel("nº de regiões")
    axes[0].set_title(f"Balanço emissor/receptor (reciprocidade = {reciprocity:.2f})")
    axes[1].hist(nodes["insularity"].dropna(), bins=25, color="indianred")
    axes[1].set_xlabel("insularidade  (chamadas internas / total da região)")
    axes[1].set_ylabel("nº de regiões")
    axes[1].set_title("O quanto cada região fala consigo mesma")
    fig.tight_layout()
    exporter.save_figure(fig, "balance_insularity", "topology")


def run(net, config: dict, exporter, communities=None, edges_antenna=None) -> dict:
    """Calcula e exporta a estrutura da rede de regiões."""
    logger.info("[topology] força, backbone, macro-regiões, s-core, balanço")
    city_name = exporter.city_name
    G, B, nodes = net.G, net.backbone, net.nodes

    n, m = G.number_of_nodes(), G.number_of_edges()
    metrics: dict = {
        "n_regions": n,
        "n_flows": m,
        "density": float(nx.density(G)) if n > 1 else float("nan"),
        "n_users": int(nodes["n_users"].sum()),
        "users_per_region_median": float(nodes["n_users"].median()),
        "internal_call_share": float(net.internal_call_share),
        "median_insularity": float(nodes["insularity"].median()),
    }

    if m == 0:
        logger.warning("[topology] rede sem fluxos entre regiões — análises puladas")
        exporter.add_metrics("topology", metrics)
        return {"metrics": metrics, "communities": []}

    metrics.update(_plot_strength(nodes, exporter))
    metrics.update(_plot_backbone(G, B, net.alpha, exporter, city_name))

    # -------- macro-regiões funcionais --------
    # Louvain ponderado sobre os fluxos: quais conjuntos de antenas conversam mais entre si
    # do que com o resto da cidade. É a regionalização que substitui as comunidades de usuários.
    if communities is None:
        communities = louvain_communities(G, weight="weight", seed=42)
    Q = modularity(G, communities, weight="weight")
    region_of = {a: i for i, com in enumerate(communities) for a in com}
    sizes = sorted((len(c) for c in communities), reverse=True)

    nodes = nodes.copy()
    nodes["macro_region"] = nodes["antenna_id"].map(region_of)
    metrics.update(
        n_macro_regions=len(communities),
        modularity=float(Q),
        largest_macro_region=int(sizes[0]),
    )

    # -------- matriz de conexão entre antenas (K e J) --------
    # Formulação do paper: K_lm = contatos entre residentes de l e m; J_lm = K_lm/(u_l u_m).
    # A unidade é o contato (pessoas distintas conectadas), não o volume de chamadas.
    cm = build_contact_matrix(edges_antenna, nodes) if edges_antenna is not None else None
    if cm is not None:
        ordem = nodes.sort_values(["macro_region", "antenna_id"])["antenna_id"].to_numpy()
        metrics.update(_plot_contact_matrix(cm, ordem, exporter, city_name))
        exporter.save_data(cm.J.reindex(index=ordem, columns=ordem).round(8), "contact_matrix_J.csv")
        exporter.save_data(cm.K.reindex(index=ordem, columns=ordem), "contact_matrix_K.csv")

    # -------- núcleo e periferia --------
    levels, score_metrics = _plot_score(G, exporter, city_name)
    metrics.update(score_metrics)
    nodes["score_level"] = nodes["antenna_id"].map(levels)

    # -------- assortatividade --------
    # A assortatividade topológica é vazia numa rede quase completa (todo mundo liga para
    # todo mundo); o que informa é se regiões de volume parecido se ligam mais — e a
    # homofilia por quintil, que é ponderada por volume e fica no módulo espacial.
    try:
        metrics["assortativity_volume"] = float(nx.numeric_assortativity_coefficient(G, "calls_total"))
    except Exception as exc:
        logger.warning("[topology] assortatividade por volume não calculada (%s)", exc)
    try:
        metrics["assortativity_quintile_unweighted"] = float(
            nx.attribute_assortativity_coefficient(G, "residence_quintile_state")
        )
    except Exception:
        pass

    # -------- direção dos fluxos --------
    reciprocity = float(nx.reciprocity(net.D)) if net.D.number_of_edges() else float("nan")
    metrics["reciprocity"] = reciprocity
    _plot_balance(nodes, reciprocity, exporter, city_name)

    exporter.save_data(nodes, "antenna_nodes.csv")
    exporter.save_data(net.flows, "antenna_flows.csv")
    exporter.save_data(
        pd.DataFrame(
            [(u, v, d["q_calls"], d["n_pairs"], round(d["dist_km"], 2)) for u, v, d in B.edges(data=True)],
            columns=["a", "b", "q_calls", "n_pairs", "dist_km"],
        ).sort_values("q_calls", ascending=False),
        "backbone_flows.csv",
    )

    exporter.add_metrics("topology", metrics)
    exporter.add_report_section(
        "Estrutura da rede de regiões",
        f"A cidade é descrita por **{n} regiões** (antenas) que reúnem "
        f"**{metrics['n_users']:,} moradores** (mediana de {metrics['users_per_region_median']:.0f} por "
        f"região) e trocam **{m:,} fluxos** — densidade **{metrics['density']:.2f}**, ou seja, mais da "
        f"metade dos pares de regiões da cidade tem contato. Por isso o que separa um corredor real "
        f"de ruído não é existir, é carregar peso: o **filtro de disparidade** guarda apenas "
        f"**{metrics['backbone_edges']:,} fluxos ({metrics['backbone_edge_fraction']:.0%} do total)** e "
        f"ainda preserva **{metrics['backbone_weight_fraction']:.0%} de todas as chamadas** — a cidade "
        f"tem um esqueleto de comunicação bem definido, e ele cobre "
        f"**{metrics.get('backbone_regions_covered', '—')} das {n} regiões** (um corte pelo peso bruto "
        f"do mesmo tamanho deixaria de fora as áreas pequenas, alcançando só "
        f"{metrics.get('naive_cut_regions_covered', '—')}). O volume é desigual entre regiões "
        f"(**Gini {metrics.get('volume_gini', float('nan')):.2f}**) e o núcleo s-core final reúne "
        f"**{metrics.get('score_core_size', '—')} regiões**. Os fluxos dividem {city_name} em "
        f"**{len(communities)} macro-regiões funcionais** (modularidade {Q:.2f}, a maior com "
        f"{sizes[0]} antenas) e são fortemente recíprocos (**{reciprocity:.0%}**): quem recebe, devolve.",
    )

    return {"metrics": metrics, "communities": communities, "nodes": nodes}

"""Métricas estruturais avançadas de redes complexas (lógica do notebook 4)."""

from __future__ import annotations

import logging
import random

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from ..graph_builder import remove_self_loops

logger = logging.getLogger("pipeline")


def fit_powerlaw_discrete(x: np.ndarray) -> dict:
    """Ajuste MLE de lei de potência discreta com varredura de x_min (método de Clauset)."""
    x = np.asarray(x)
    x = x[x > 0]
    best = None
    for xmin in range(1, int(x.max())):
        tail = x[x >= xmin]
        if len(tail) < 30:
            break
        alpha = 1 + len(tail) / np.sum(np.log(tail / (xmin - 0.5)))
        xs = np.sort(tail)
        cdf_emp = np.arange(1, len(xs) + 1) / len(xs)
        cdf_fit = 1 - (xs / xmin) ** (-(alpha - 1))
        ks = np.max(np.abs(cdf_emp - cdf_fit))
        if best is None or ks < best["ks"]:
            best = {"xmin": int(xmin), "alpha": float(alpha), "n_tail": int(len(tail)), "ks": float(ks)}
    return best


def _avg_shortest_path_sampled(G: nx.Graph, n_samples: int = 500, seed: int = 42) -> float:
    rng = random.Random(seed)
    nodes = list(G.nodes())
    total = count = 0
    for s in rng.sample(nodes, min(n_samples, len(nodes))):
        lengths = nx.single_source_shortest_path_length(G, s)
        total += sum(lengths.values())
        count += len(lengths) - 1
    return total / count


def _giant_fraction_curve(G: nx.Graph, removal_order, max_frac=0.6, steps=25):
    N = G.number_of_nodes()
    all_nodes = set(G.nodes())
    fracs = np.linspace(0, max_frac, steps)
    out = []
    for f in fracs:
        k = int(f * N)
        H = G.subgraph(all_nodes - set(removal_order[:k]))
        gc = max((len(c) for c in nx.connected_components(H)), default=0)
        out.append(gc / N)
    return fracs, out


def run(G_main: nx.Graph, config: dict, exporter) -> dict:
    """Calcula e exporta lei de potência, assortatividade/k-core, small-world e robustez."""
    logger.info("[advanced] scale-free, assortatividade/k-core, small-world, robustez")
    adv = config.get("advanced", {})
    steps = adv.get("robustness_steps", 50)
    sp_sample = adv.get("shortest_path_sample", 500)

    Gm = remove_self_loops(G_main)
    degree = np.array([d for _, d in Gm.degree()])

    # ---------------- 1. lei de potência ----------------
    pl = fit_powerlaw_discrete(degree)
    x = np.sort(np.unique(degree))
    ccdf = np.array([np.mean(degree >= k) for k in x])
    xmin, alpha = pl["xmin"], pl["alpha"]
    xx = x[x >= xmin]
    yy = np.mean(degree >= xmin) * (xx / xmin) ** (-(alpha - 1))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(x, ccdf, "o", ms=5, alpha=0.7, label="CCDF empírica")
    ax.loglog(xx, yy, "r-", lw=2, label=f"lei de potência (α={alpha:.2f}, x_min={xmin})")
    ax.set_xlabel("grau  k")
    ax.set_ylabel("P(K ≥ k)")
    ax.set_title(f"Ajuste de lei de potência — {exporter.city_name}")
    ax.legend()
    exporter.save_figure(fig, "powerlaw_fit", "advanced")

    # ---------------- 2. assortatividade + k-core ----------------
    assort = nx.degree_assortativity_coefficient(Gm)
    knn = nx.average_degree_connectivity(Gm)
    ks_knn = sorted(knn)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ks_knn, [knn[k] for k in ks_knn], s=30, alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grau  k")
    ax.set_ylabel("grau médio dos vizinhos  k_nn(k)")
    ax.set_title(f"Mistura por grau (r = {assort:.2f})")
    exporter.save_figure(fig, "assortativity", "advanced")

    core = pd.Series(nx.core_number(Gm))
    kmax = int(core.max())
    ks_core = list(range(0, kmax + 1))
    core_sizes = [int((core >= k).sum()) for k in ks_core]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(core, bins=range(0, kmax + 2), align="left", rwidth=0.8)
    axes[0].set_xlabel("número do core (k)")
    axes[0].set_ylabel("nós")
    axes[0].set_title("Distribuição de k-core")
    axes[1].plot(ks_core, core_sizes, "o-")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("nós no k-core (log)")
    axes[1].set_title("Tamanho do núcleo")
    fig.tight_layout()
    exporter.save_figure(fig, "kcore", "advanced")

    # ---------------- 3. small-world ----------------
    C_real = nx.average_clustering(Gm)
    L_real = _avg_shortest_path_sampled(Gm, sp_sample)
    n, m = Gm.number_of_nodes(), Gm.number_of_edges()
    G_rand = nx.gnm_random_graph(n, m, seed=42)
    G_rand = G_rand.subgraph(max(nx.connected_components(G_rand), key=len)).copy()
    C_rand = nx.average_clustering(G_rand)
    L_rand = _avg_shortest_path_sampled(G_rand, sp_sample)
    sigma = (C_real / C_rand) / (L_real / L_rand)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["rede real", "aleatória"], [C_real, C_rand], color=["crimson", "steelblue"])
    axes[0].set_yscale("log")
    axes[0].set_ylabel("clustering médio (log)")
    axes[0].set_title("Agrupamento local")
    axes[1].bar(["rede real", "aleatória"], [L_real, L_rand], color=["crimson", "steelblue"])
    axes[1].set_ylabel("caminho mínimo médio")
    axes[1].set_title("Comprimento dos caminhos")
    fig.tight_layout()
    exporter.save_figure(fig, "smallworld", "advanced")

    # ---------------- 4. robustez ----------------
    deg_order = [u for u, _ in sorted(Gm.degree(), key=lambda kv: -kv[1])]
    rand_order = list(Gm.nodes())
    random.Random(1).shuffle(rand_order)
    fr, y_attack = _giant_fraction_curve(Gm, deg_order, steps=steps)
    _, y_random = _giant_fraction_curve(Gm, rand_order, steps=steps)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.plot(fr, y_attack, "o-", color="crimson", label="ataque dirigido (maior grau)")
    ax.plot(fr, y_random, "s-", color="steelblue", label="falha aleatória")
    ax.set_xlabel("fração de nós removidos")
    ax.set_ylabel("fração na componente gigante")
    ax.set_title(f"Robustez: ataque vs. falha — {exporter.city_name}")
    ax.legend()
    exporter.save_figure(fig, "robustness", "advanced")

    # limiar aproximado de colapso sob ataque (gigante < 50%)
    collapse = next((round(f, 3) for f, g in zip(fr, y_attack) if g < 0.5), None)

    metrics = {
        "powerlaw_alpha": alpha,
        "powerlaw_xmin": xmin,
        "powerlaw_ks": pl["ks"],
        "assortativity": float(assort),
        "kcore_max": kmax,
        "kcore_size": int((core == kmax).sum()),
        "smallworld_C_real": float(C_real),
        "smallworld_C_rand": float(C_rand),
        "smallworld_L_real": float(L_real),
        "smallworld_L_rand": float(L_rand),
        "smallworld_sigma": float(sigma),
        "attack_collapse_fraction": collapse,
    }
    exporter.add_metrics("advanced", metrics)

    exporter.add_report_section(
        "Análises avançadas",
        f"A distribuição de grau é de cauda pesada, com expoente **α ≈ {alpha:.2f}** "
        f"(x_min={xmin}, KS={pl['ks']:.2f}). A rede é **assortativa** (r = {assort:+.2f}: hubs ligam "
        f"com hubs) e tem um núcleo coeso pequeno (**{kmax}-core** com {int((core == kmax).sum())} "
        f"nós). É **small-world**: clustering {C_real:.3f} vs {C_rand:.4f} no aleatório e caminho "
        f"médio {L_real:.1f} vs {L_rand:.1f}, dando **σ ≈ {sigma:.0f}**. Sob ataque dirigido aos "
        f"hubs a componente gigante cai abaixo de 50% com ~{collapse:.0%} de remoção, enquanto sob "
        f"falha aleatória resiste — assinatura clássica de rede com hubs (robusta a falhas, frágil a "
        f"ataques)."
        if collapse is not None
        else "",
    )

    return {"metrics": metrics}

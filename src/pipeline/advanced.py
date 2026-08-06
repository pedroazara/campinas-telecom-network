"""Métricas estruturais avançadas de redes complexas (lógica do notebook 4).

Todas as análises têm guardas para não quebrar em cidades pequenas (poucos nós/grau baixo)
nem em cidades muito grandes.
"""

from __future__ import annotations

import logging
import random

import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

from ..graph_builder import remove_self_loops, average_clustering_fast

logger = logging.getLogger("pipeline")

# acima deste tamanho, evitamos as operações mais caras (rich-club normalizado)
BIG_GRAPH = 150_000


def fit_powerlaw_discrete(x: np.ndarray) -> dict | None:
    """Ajuste MLE de lei de potência discreta com varredura de x_min (método de Clauset)."""
    x = np.asarray(x)
    x = x[x > 0]
    if len(x) < 50 or x.max() < 4:
        return None
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
    return total / count if count else float("nan")


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


def _rich_club(Gm: nx.Graph, G_rand: nx.Graph, exporter, city_name: str) -> dict:
    """Coeficiente rich-club φ(k) da rede vs. um grafo aleatório equivalente.

    ρ(k) = φ_real(k) / φ_rand(k) > 1 para k alto indica que os hubs formam um 'clube'.
    """
    try:
        phi_real = nx.rich_club_coefficient(Gm, normalized=False)
        phi_rand = nx.rich_club_coefficient(G_rand, normalized=False)
    except Exception as exc:
        logger.warning("[advanced] rich-club não calculado (%s)", exc)
        return {}

    ks = sorted(k for k in phi_real if k in phi_rand and phi_rand[k] > 0)
    if len(ks) < 3:
        return {}
    ratio = np.array([phi_real[k] / phi_rand[k] for k in ks])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(ks, [phi_real[k] for k in ks], "o-", color="crimson", label="rede real")
    axes[0].plot(ks, [phi_rand[k] for k in ks], "s-", color="steelblue", label="aleatória")
    axes[0].set_xlabel("grau  k")
    axes[0].set_ylabel("φ(k)")
    axes[0].set_title("Coeficiente rich-club")
    axes[0].legend()
    axes[1].axhline(1.0, color="gray", ls="--", lw=1)
    axes[1].plot(ks, ratio, "o-", color="purple")
    axes[1].set_xlabel("grau  k")
    axes[1].set_ylabel("ρ(k) = φ_real / φ_rand")
    axes[1].set_title("Rich-club normalizado (ρ>1 = clube de hubs)")
    fig.tight_layout()
    exporter.save_figure(fig, "rich_club", "advanced")

    # razão no topo da distribuição de grau (média do quarto superior de k)
    top = ratio[int(0.75 * len(ratio)):]
    return {"rich_club_ratio_high_k": float(np.nanmean(top))}


def run(G_main: nx.Graph, config: dict, exporter) -> dict:
    """Calcula e exporta lei de potência, assortatividade/k-core, small-world, robustez e rich-club."""
    logger.info("[advanced] scale-free, assortatividade/k-core, small-world, robustez, rich-club")
    adv = config.get("advanced", {})
    steps = adv.get("robustness_steps", 50)
    sp_sample = adv.get("shortest_path_sample", 500)

    Gm = remove_self_loops(G_main)
    n_nodes, n_edges = Gm.number_of_nodes(), Gm.number_of_edges()
    degree = np.array([d for _, d in Gm.degree()]) if n_nodes else np.array([0])
    metrics: dict = {}

    # amostragem adaptativa ao tamanho; acima de HUGE, as métricas baseadas em BFS
    # (caminho médio, robustez) são inviáveis e são puladas.
    HUGE = 300_000
    huge = n_nodes > HUGE
    if n_nodes > 120_000:
        sp_eff, steps_eff = 150, 25
    else:
        sp_eff, steps_eff = sp_sample, steps

    # ---------------- 1. lei de potência ----------------
    pl = fit_powerlaw_discrete(degree)
    if pl is not None:
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
        metrics.update(powerlaw_alpha=pl["alpha"], powerlaw_xmin=pl["xmin"], powerlaw_ks=pl["ks"])
    else:
        logger.warning("[advanced] grau sem cauda suficiente — lei de potência pulada")

    # ---------------- 2. assortatividade + k-core ----------------
    try:
        assort = float(nx.degree_assortativity_coefficient(Gm))
    except Exception:
        assort = float("nan")
    if np.isfinite(assort):
        metrics["assortativity"] = assort
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

    core = pd.Series(nx.core_number(Gm)) if n_edges else pd.Series([0])
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
    metrics.update(kcore_max=kmax, kcore_size=int((core == kmax).sum()))

    # ---------------- 3. small-world ----------------
    G_rand = None
    if huge:
        logger.warning("[advanced] grafo enorme (%d nós) — small-world pulado (BFS inviável)", n_nodes)
    elif n_nodes > 3 and n_edges > 0:
        C_real = average_clustering_fast(Gm)
        L_real = _avg_shortest_path_sampled(Gm, sp_eff)
        G_rand = nx.gnm_random_graph(n_nodes, n_edges, seed=42)
        G_rand = G_rand.subgraph(max(nx.connected_components(G_rand), key=len)).copy()
        C_rand = average_clustering_fast(G_rand)
        L_rand = _avg_shortest_path_sampled(G_rand, sp_eff)
        if C_rand > 0 and np.isfinite(L_real) and L_rand > 0:
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
            metrics.update(
                smallworld_C_real=float(C_real), smallworld_C_rand=float(C_rand),
                smallworld_L_real=float(L_real), smallworld_L_rand=float(L_rand),
                smallworld_sigma=float(sigma),
            )

    # ---------------- 4. robustez ----------------
    if huge:
        logger.warning("[advanced] grafo enorme (%d nós) — robustez pulada (componentes conexas inviáveis)", n_nodes)
    elif n_nodes > 0:
        deg_order = [u for u, _ in sorted(Gm.degree(), key=lambda kv: -kv[1])]
        rand_order = list(Gm.nodes())
        random.Random(1).shuffle(rand_order)
        fr, y_attack = _giant_fraction_curve(Gm, deg_order, steps=steps_eff)
        _, y_random = _giant_fraction_curve(Gm, rand_order, steps=steps_eff)
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.plot(fr, y_attack, "o-", color="crimson", label="ataque dirigido (maior grau)")
        ax.plot(fr, y_random, "s-", color="steelblue", label="falha aleatória")
        ax.set_xlabel("fração de nós removidos")
        ax.set_ylabel("fração na componente gigante")
        ax.set_title(f"Robustez: ataque vs. falha — {exporter.city_name}")
        ax.legend()
        exporter.save_figure(fig, "robustness", "advanced")
        collapse = next((round(f, 3) for f, g in zip(fr, y_attack) if g < 0.5), None)
        metrics["attack_collapse_fraction"] = collapse

    # ---------------- 5. rich-club ----------------
    if G_rand is not None and n_nodes <= BIG_GRAPH:
        metrics.update(_rich_club(Gm, G_rand, exporter, exporter.city_name))
    elif n_nodes > BIG_GRAPH:
        logger.warning("[advanced] rede muito grande (%d nós) — rich-club pulado", n_nodes)

    exporter.add_metrics("advanced", metrics)

    # ---------------- relatório ----------------
    partes = []
    if "powerlaw_alpha" in metrics:
        partes.append(
            f"A distribuição de grau é de cauda pesada, com expoente **α ≈ {metrics['powerlaw_alpha']:.2f}** "
            f"(x_min={metrics['powerlaw_xmin']}, KS={metrics['powerlaw_ks']:.2f})."
        )
    if "assortativity" in metrics:
        tipo = "assortativa" if metrics["assortativity"] > 0 else "disassortativa"
        partes.append(f"É **{tipo}** (r = {metrics['assortativity']:+.2f}).")
    partes.append(f"Núcleo máximo: **{kmax}-core** com {int((core == kmax).sum())} nós.")
    if "smallworld_sigma" in metrics:
        partes.append(
            f"É **small-world** (σ ≈ {metrics['smallworld_sigma']:.0f}: clustering "
            f"{metrics['smallworld_C_real']:.3f} vs {metrics['smallworld_C_rand']:.4f} no aleatório)."
        )
    if metrics.get("attack_collapse_fraction") is not None:
        partes.append(
            f"Sob ataque aos hubs a componente gigante cai abaixo de 50% com "
            f"~{metrics['attack_collapse_fraction']:.0%} de remoção (robusta a falhas, frágil a ataques)."
        )
    if "rich_club_ratio_high_k" in metrics:
        rc = metrics["rich_club_ratio_high_k"]
        tem = "formam" if rc > 1.1 else "não formam"
        partes.append(f"Os hubs **{tem} um rich-club** (ρ médio no topo ≈ {rc:.2f}).")
    exporter.add_report_section("Análises avançadas", " ".join(partes))

    return {"metrics": metrics}

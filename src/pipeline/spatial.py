"""Análise espacial e socioeconômica da rede (lógica do notebook 3)."""

from __future__ import annotations

import ast
import logging

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely import wkb
from shapely.geometry import Point, Polygon, LineString, box
from scipy.spatial import Voronoi
from networkx.algorithms.community import louvain_communities

from ..graph_builder import build_edges_graph, build_user_antenna_map

logger = logging.getLogger("pipeline")


# --------------------------------------------------------------------------- mapas
def _add_basemap(ax, config) -> None:
    """Adiciona o mapa de fundo (contextily) se habilitado e disponível."""
    if not config.get("spatial", {}).get("download_basemap", True):
        return
    try:
        import contextily as ctx

        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except Exception as exc:  # offline / sem tiles: segue sem o basemap
        logger.warning("basemap não baixado (%s); seguindo sem tiles", exc)


def _build_gdf(antennas_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Converte a geometria WKB das antenas em pontos georreferenciados (EPSG:3857)."""
    ant = antennas_df.copy()
    geom = ant["residence_geometry"].apply(
        lambda s: wkb.loads(ast.literal_eval(s)) if isinstance(s, str) else s
    )
    ant["lon"] = geom.apply(lambda p: p.x)
    ant["lat"] = geom.apply(lambda p: p.y)
    gdf = gpd.GeoDataFrame(
        ant, geometry=gpd.points_from_xy(ant["lon"], ant["lat"]), crs="EPSG:4326"
    ).to_crs(epsg=3857)
    if "residence_geometry" in gdf.columns:
        gdf = gdf.drop(columns="residence_geometry")
    return gdf


def _voronoi_finite_polygons_2d(vor, radius):
    """Reconstrói células de Voronoi finitas (mesma rotina dos notebooks)."""
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
    for p1, region_idx in enumerate(vor.point_region):
        vertices = vor.regions[region_idx]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]
        new_regions.append(new_region.tolist())
    return new_regions, np.asarray(new_vertices)


def _build_voronoi(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    vor = Voronoi(coords)
    radius = (coords.max(axis=0) - coords.min(axis=0)).max() * 2
    regions, vertices = _voronoi_finite_polygons_2d(vor, radius)
    polys = [Polygon(vertices[r]) for r in regions]
    vor_gdf = gpd.GeoDataFrame(gdf.drop(columns="geometry"), geometry=polys, crs=gdf.crs)
    xmin, ymin, xmax, ymax = gdf.total_bounds
    clip = box(xmin - 5000, ymin - 5000, xmax + 5000, ymax + 5000)
    vor_gdf["geometry"] = vor_gdf.geometry.intersection(clip)
    return vor_gdf


def _random_point_in_polygon(poly, rng, fallback=None, max_tries=200):
    """Sorteia um ponto aproximadamente uniforme dentro de um polígono (rejection sampling)."""
    if poly is None or poly.is_empty:
        return fallback
    minx, miny, maxx, maxy = poly.bounds
    for _ in range(max_tries):
        p = Point(rng.uniform(minx, maxx), rng.uniform(miny, maxy))
        if poly.contains(p):
            return p
    c = poly.centroid
    return c if (c is not None and not c.is_empty) else fallback


def _plot_community_network(G_main, gdf, vor_gdf, community_df, community_sizes,
                            user_antenna, edges_graph, config, exporter, city_name):
    """Desenha o recorte das maiores comunidades: usuários posicionados dentro da célula de
    Voronoi da sua antena, com as fronteiras das regiões e as arestas da rede."""
    rng = np.random.default_rng(42)
    n_show = int(min(
        config.get("spatial", {}).get("network_map_communities", 12),
        community_sizes["community"].nunique(),
    ))
    largest = community_sizes.nsmallest(n_show, "community_rank")["community"]
    users = set(community_df.loc[community_df["community"].isin(largest), "user_id"])

    top_edges = edges_graph[
        edges_graph["source"].isin(users) & edges_graph["target"].isin(users)
    ].rename(columns={"source": "id_emisor", "target": "id_receiver"}).copy()

    nodes = pd.DataFrame({"user_id": sorted(users)})
    nodes["antenna_id"] = nodes["user_id"].map(user_antenna)
    nodes = nodes.dropna(subset=["antenna_id"]).merge(
        community_df[["user_id", "community_rank"]], on="user_id", how="left"
    )

    antenna_point = gdf.set_index("antenna_id").geometry
    antenna_poly = vor_gdf.set_index("antenna_id").geometry if vor_gdf is not None else None
    geoms = []
    for aid in nodes["antenna_id"]:
        poly = antenna_poly.get(aid) if antenna_poly is not None else None
        geoms.append(_random_point_in_polygon(poly, rng, fallback=antenna_point.get(aid)))
    nodes["geometry"] = geoms
    nodes = nodes.dropna(subset=["geometry"])
    nodes_gdf = gpd.GeoDataFrame(nodes, geometry="geometry", crs=gdf.crs)

    coords = nodes_gdf.set_index("user_id").geometry.to_dict()
    lines = [
        LineString([coords[s], coords[t]])
        for s, t in zip(top_edges["id_emisor"], top_edges["id_receiver"])
        if s in coords and t in coords
    ]
    edges_gdf = gpd.GeoDataFrame(geometry=lines, crs=gdf.crs)

    fig, ax = plt.subplots(figsize=(11, 11))
    if vor_gdf is not None:
        vor_gdf.boundary.plot(ax=ax, color="dimgray", linewidth=0.6, alpha=0.5)
    if len(edges_gdf):
        edges_gdf.plot(ax=ax, color="steelblue", linewidth=0.4, alpha=0.15)
    nodes_gdf.plot(ax=ax, column="community_rank", cmap="tab20", categorical=True,
                   markersize=6, legend=False)
    gdf.plot(ax=ax, color="black", markersize=18, alpha=0.7)
    _add_basemap(ax, config)
    ax.set_axis_off()
    ax.set_title(f"Recorte das {n_show} maiores comunidades — {city_name}")
    exporter.save_figure(fig, "community_network", "spatial")


def _community_spatial_concentration(community_df, community_sizes, user_antenna, gdf,
                                     exporter, city_name):
    """Quantifica o quanto cada comunidade é territorialmente concentrada ou dispersa."""
    cn = community_df.copy()
    cn["antenna_id"] = cn["user_id"].map(user_antenna)
    cn = cn.merge(gdf[["antenna_id", "geometry"]], on="antenna_id", how="left").dropna(
        subset=["antenna_id", "geometry"]
    )
    cn["x"] = cn["geometry"].apply(lambda p: p.x)
    cn["y"] = cn["geometry"].apply(lambda p: p.y)

    counts = cn.groupby(["community", "antenna_id"]).size().rename("n").reset_index()
    total = counts.groupby("community")["n"].transform("sum")
    counts["p"] = counts["n"] / total
    entropy = (
        counts.assign(plogp=lambda d: d["p"] * np.log(d["p"]))
        .groupby("community")["plogp"].sum().mul(-1).rename("antenna_entropy")
    )
    dominant = (
        counts.sort_values(["community", "n"], ascending=[True, False])
        .drop_duplicates("community").set_index("community")["p"].rename("dominant_antenna_share")
    )
    n_antennas = cn.groupby("community")["antenna_id"].nunique().rename("n_antennas")

    centroids = cn.groupby("community").agg(cx=("x", "mean"), cy=("y", "mean"))
    cn = cn.merge(centroids, on="community", how="left")
    cn["radius_km"] = np.sqrt((cn["x"] - cn["cx"]) ** 2 + (cn["y"] - cn["cy"]) ** 2) / 1000
    radius = cn.groupby("community")["radius_km"].mean().rename("mean_radius_km")

    summary = (
        community_sizes.set_index("community")
        .join(n_antennas).join(dominant).join(entropy).join(radius)
    )
    summary["effective_antennas"] = np.exp(summary["antenna_entropy"])
    summary = summary.sort_values("community_size", ascending=False).reset_index()
    exporter.save_data(summary, "community_spatial.csv")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].hist(summary["dominant_antenna_share"], bins=20)
    axes[0].set_xlabel("share da antena dominante")
    axes[0].set_ylabel("nº de comunidades")
    axes[0].set_title("Concentração em uma antena")
    axes[1].scatter(summary["community_size"], summary["mean_radius_km"], s=20, alpha=0.7)
    axes[1].set_xlabel("tamanho da comunidade")
    axes[1].set_ylabel("raio médio (km)")
    axes[1].set_title("Tamanho vs. dispersão espacial")
    fig.tight_layout()
    exporter.save_figure(fig, "community_spatial", "spatial")

    exporter.add_metrics("spatial", {
        "median_dominant_antenna_share": float(summary["dominant_antenna_share"].median()),
        "median_effective_antennas": float(summary["effective_antennas"].median()),
        "median_community_radius_km": float(summary["mean_radius_km"].median()),
    })
    return summary


# --------------------------------------------------------------------------- run
def run(G_main: nx.Graph, antennas_df: pd.DataFrame, edges_antenna_df: pd.DataFrame,
        config: dict, exporter) -> dict:
    """Calcula e exporta Voronoi, homofilia socioeconômica, decaimento, rede de antenas e hubs."""
    logger.info("[spatial] Voronoi, homofilia, decaimento, rede de antenas, hubs")
    city_name = exporter.city_name

    gdf = _build_gdf(antennas_df)
    edges_graph = build_edges_graph(edges_antenna_df)
    user_antenna = build_user_antenna_map(edges_antenna_df)

    # comunidades (Louvain) — usadas no mapa da rede e na concentração espacial
    communities = louvain_communities(G_main, weight="weight", seed=42)
    community_df = pd.DataFrame(
        [{"user_id": u, "community": i} for i, com in enumerate(communities) for u in com]
    )
    community_sizes = community_df.groupby("community").size().rename("community_size").reset_index()
    community_sizes["community_rank"] = (
        community_sizes["community_size"].rank(method="first", ascending=False).astype(int)
    )
    community_df = community_df.merge(community_sizes, on="community", how="left")

    vor_gdf = _build_voronoi(gdf) if config.get("spatial", {}).get("voronoi", True) else None

    # -------- Voronoi por quintil --------
    if vor_gdf is not None:
        fig, ax = plt.subplots(figsize=(11, 11))
        vor_gdf.plot(
            column="residence_quintile_state", cmap="RdYlGn", categorical=True,
            legend=True, edgecolor="black", linewidth=0.5, alpha=0.3, ax=ax,
        )
        gdf.plot(ax=ax, color="black", markersize=5)
        _add_basemap(ax, config)
        ax.set_axis_off()
        ax.set_title(f"Regiões de influência das antenas por quintil — {city_name}")
        exporter.save_figure(fig, "voronoi_map", "spatial")

    # -------- visualização da rede (recorte das maiores comunidades) --------
    _plot_community_network(
        G_main, gdf, vor_gdf, community_df, community_sizes, user_antenna,
        edges_graph, config, exporter, city_name,
    )

    # -------- concentração espacial das comunidades --------
    community_spatial = _community_spatial_concentration(
        community_df, community_sizes, user_antenna, gdf, exporter, city_name
    )

    # -------- homofilia socioeconômica --------
    quintil_por_antena = gdf.set_index("antenna_id")["residence_quintile_state"]
    user_quintile = user_antenna.map(quintil_por_antena)

    giant_nodes = list(G_main.nodes())
    node_q = user_quintile.reindex(giant_nodes)
    eh = edges_graph[
        edges_graph["source"].isin(giant_nodes) & edges_graph["target"].isin(giant_nodes)
    ].copy()
    eh["q_source"] = eh["source"].map(user_quintile)
    eh["q_target"] = eh["target"].map(user_quintile)
    eh = eh.dropna(subset=["q_source", "q_target"])

    obs_same = float((eh["q_source"] == eh["q_target"]).mean())
    codes, _ = pd.factorize(node_q)
    pos = {nd: i for i, nd in enumerate(giant_nodes)}
    si = eh["source"].map(pos).to_numpy()
    ti = eh["target"].map(pos).to_numpy()
    rng = np.random.default_rng(42)
    null_same = [float((perm[si] == perm[ti]).mean())
                 for perm in (rng.permutation(codes) for _ in range(100))]
    null_mean = float(np.mean(null_same))
    ratio = obs_same / null_mean

    ordem = ["q1", "q2", "q3", "q4", "q5"]
    mix = pd.crosstab(eh["q_source"], eh["q_target"]).reindex(index=ordem, columns=ordem, fill_value=0)
    mix_sym = mix + mix.T
    mix_norm = mix_sym.div(mix_sym.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mix_norm, annot=True, fmt=".2f", cmap="rocket_r", ax=ax)
    ax.set_title(f"Matriz de mistura por quintil — {city_name}")
    ax.set_xlabel("quintil do outro extremo")
    ax.set_ylabel("quintil de origem")
    exporter.save_figure(fig, "homophily_matrix", "spatial")

    # -------- decaimento com a distância --------
    dist_ok = edges_graph.dropna(subset=["residence_distance_km"]).copy()
    dmax = dist_ok["residence_distance_km"].quantile(0.99)
    bins = np.linspace(0, dmax, 16)
    dist_ok["faixa"] = pd.cut(dist_ok["residence_distance_km"], bins)
    decay = (
        dist_ok.groupby("faixa", observed=True)
        .agg(n=("q_calls", "size"), mean_q=("q_calls", "mean"))
        .reset_index()
    )
    decay["dist"] = decay["faixa"].apply(lambda i: i.mid)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(decay["dist"], decay["mean_q"], "o-")
    axes[0].set_xlabel("distância residencial (km)")
    axes[0].set_ylabel("chamadas médias por aresta")
    axes[0].set_title("Intensidade vs. distância")
    axes[1].loglog(decay["dist"], decay["mean_q"], "o")
    axes[1].set_xlabel("distância (km)")
    axes[1].set_ylabel("chamadas médias por aresta")
    axes[1].set_title("Mesma relação (log-log)")
    fig.tight_layout()
    exporter.save_figure(fig, "distance_decay", "spatial")

    # -------- rede agregada entre antenas --------
    ea = edges_antenna_df.dropna(subset=["emissor_antenna_id", "receptor_antenna_id"]).copy()
    ea["a"] = np.minimum(ea["emissor_antenna_id"], ea["receptor_antenna_id"])
    ea["b"] = np.maximum(ea["emissor_antenna_id"], ea["receptor_antenna_id"])
    ea = ea[ea["a"] != ea["b"]]
    antenna_edges = ea.groupby(["a", "b"], as_index=False).agg(
        q_calls=("q_calls", "sum"), n_pares=("q_calls", "size")
    )
    GA = nx.from_pandas_edgelist(antenna_edges, "a", "b", edge_attr=["q_calls", "n_pares"])
    exporter.save_data(
        antenna_edges.sort_values("q_calls", ascending=False), "antenna_flows.csv"
    )

    ant_xy = gdf.set_index("antenna_id").geometry
    flows = antenna_edges[
        antenna_edges["a"].isin(ant_xy.index) & antenna_edges["b"].isin(ant_xy.index)
    ].copy()
    flows["geometry"] = [LineString([ant_xy[a], ant_xy[b]]) for a, b in zip(flows["a"], flows["b"])]
    flows_gdf = gpd.GeoDataFrame(flows, geometry="geometry", crs=gdf.crs)
    strong = flows_gdf.sort_values("q_calls", ascending=False).head(400)
    lw = 0.2 + 4.0 * strong["q_calls"] / strong["q_calls"].max()
    fig, ax = plt.subplots(figsize=(11, 11))
    strong.plot(ax=ax, color="crimson", linewidth=lw, alpha=0.35)
    gdf.plot(ax=ax, color="black", markersize=12)
    _add_basemap(ax, config)
    ax.set_axis_off()
    ax.set_title(f"Fluxo de chamadas entre antenas — {city_name}")
    exporter.save_figure(fig, "antenna_network", "spatial")

    # -------- hubs no mapa --------
    deg = dict(G_main.degree())
    hubs = pd.DataFrame({"user_id": list(G_main.nodes())})
    hubs["grau"] = hubs["user_id"].map(deg)
    hubs = hubs.nlargest(200, "grau")
    hubs["antenna_id"] = hubs["user_id"].map(user_antenna)
    hubs = hubs.merge(gdf[["antenna_id", "geometry"]], on="antenna_id", how="left").dropna(
        subset=["geometry"]
    )
    hubs_gdf = gpd.GeoDataFrame(hubs, geometry="geometry", crs=gdf.crs)
    fig, ax = plt.subplots(figsize=(11, 11))
    gdf.plot(ax=ax, color="lightgray", markersize=8)
    hubs_gdf.plot(ax=ax, column="grau", cmap="plasma", markersize=40, legend=True, alpha=0.85)
    _add_basemap(ax, config)
    ax.set_axis_off()
    ax.set_title(f"Antenas dos 200 maiores hubs — {city_name}")
    exporter.save_figure(fig, "hubs_map", "spatial")

    metrics = {
        "n_antennas": int(gdf["antenna_id"].nunique()),
        "homophily_observed": obs_same,
        "homophily_null": null_mean,
        "homophily_ratio": float(ratio),
        "antenna_network_nodes": GA.number_of_nodes(),
        "antenna_network_edges": GA.number_of_edges(),
        "antenna_network_density": float(nx.density(GA)),
    }
    exporter.add_metrics("spatial", metrics)

    exporter.add_report_section(
        "Análise espacial e socioeconômica",
        f"Os usuários se distribuem por **{gdf['antenna_id'].nunique()} antenas** (regiões de "
        f"Voronoi). Há **homofilia socioeconômica**: **{obs_same:.0%}** das chamadas ligam pessoas "
        f"do mesmo quintil, contra **{null_mean:.0%}** esperado ao acaso — cerca de **{ratio:.1f}×**. "
        f"A intensidade das chamadas cai com a distância residencial (efeito de gravidade espacial). "
        f"Agregada por antena, a rede tem **{GA.number_of_nodes()} regiões** e "
        f"**{GA.number_of_edges():,} fluxos** (densidade {nx.density(GA):.2f}), revelando os "
        f"principais corredores de comunicação da cidade.",
    )

    exporter.add_report_section(
        "Estrutura espacial das comunidades",
        f"O recorte das maiores comunidades sobre o mapa mostra o quanto os grupos são "
        f"territoriais. Em média, cada comunidade se concentra em "
        f"**{community_spatial['effective_antennas'].median():.1f} antenas efetivas** "
        f"(share mediano da antena dominante = "
        f"{community_spatial['dominant_antenna_share'].median():.0%}), com raio médio de "
        f"**{community_spatial['mean_radius_km'].median():.1f} km**. Quanto maior a concentração e "
        f"menor o raio, mais a comunidade corresponde a uma região específica da cidade; "
        f"comunidades espalhadas indicam grupos sociais que atravessam o território.",
    )

    return {"metrics": metrics}

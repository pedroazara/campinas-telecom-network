"""Análise espacial e socioeconômica da rede de regiões.

Com a antena como nó, o mapa deixa de ser pano de fundo: cada célula de Voronoi **é** um nó
e cada linha desenhada **é** uma aresta. Isso abre o que a rede de usuários não permitia —
modelo de gravidade, resíduos de fluxo, insularidade e balanço emissor/receptor por região.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from matplotlib.colors import LogNorm
from shapely.geometry import Polygon, LineString, box
from scipy.spatial import Voronoi

from ..antenna import fit_gravity_model
from ..boundary import load_boundary

logger = logging.getLogger("pipeline")

QUINTIS = ["q1", "q2", "q3", "q4", "q5"]


# --------------------------------------------------------------------------- mapas base
def _add_basemap(ax, config) -> None:
    """Adiciona o mapa de fundo (contextily) se habilitado e disponível."""
    if not config.get("spatial", {}).get("download_basemap", True):
        return
    try:
        import contextily as ctx

        ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    except Exception as exc:  # offline / sem tiles: segue sem o basemap
        logger.warning("basemap não baixado (%s); seguindo sem tiles", exc)


def _build_gdf(nodes: pd.DataFrame) -> gpd.GeoDataFrame:
    """Nós georreferenciados em EPSG:3857 (métrico, compatível com os tiles)."""
    valid = nodes.dropna(subset=["lon", "lat"])
    return gpd.GeoDataFrame(
        valid, geometry=gpd.points_from_xy(valid["lon"], valid["lat"]), crs="EPSG:4326"
    ).to_crs(epsg=3857)


def _voronoi_finite_polygons_2d(vor, radius):
    """Reconstrói células de Voronoi finitas a partir das regiões infinitas do scipy."""
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


def _build_voronoi(gdf: gpd.GeoDataFrame, boundary: gpd.GeoSeries | None = None) -> gpd.GeoDataFrame:
    """Células de Voronoi das antenas, recortadas pelo limite da cidade.

    O diagrama de Voronoi é infinito nas bordas. Com ``boundary``, cada célula é cortada
    pela área urbana funcional e passa a ser a fatia de território de fato atribuída à
    antena; sem ele, o recorte cai num retângulo com 5 km de folga — o comportamento
    anterior, que estica as células de borda até um limite arbitrário.
    """
    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    vor = Voronoi(coords)
    radius = (coords.max(axis=0) - coords.min(axis=0)).max() * 2
    regions, vertices = _voronoi_finite_polygons_2d(vor, radius)
    polys = [Polygon(vertices[r]) for r in regions]
    vor_gdf = gpd.GeoDataFrame(gdf.drop(columns="geometry"), geometry=polys, crs=gdf.crs)

    if boundary is not None and len(boundary):
        clip = boundary.to_crs(gdf.crs).union_all()
    else:
        xmin, ymin, xmax, ymax = gdf.total_bounds
        clip = box(xmin - 5000, ymin - 5000, xmax + 5000, ymax + 5000)

    vor_gdf["geometry"] = vor_gdf.geometry.intersection(clip)
    vor_gdf["area_km2"] = vor_gdf.geometry.to_crs(_equal_area_crs(gdf)).area / 1e6
    return vor_gdf


def _equal_area_crs(gdf: gpd.GeoDataFrame) -> str:
    """CRS equivalente em área para medir células — o EPSG:3857 infla a área com a latitude."""
    lon, lat = gdf.to_crs(epsg=4326).union_all().centroid.coords[0]
    return f"+proj=laea +lat_0={lat:.4f} +lon_0={lon:.4f} +datum=WGS84 +units=m +no_defs"


def _plot_boundary(ax, boundary: gpd.GeoSeries | None, crs) -> None:
    """Desenha o contorno da cidade — a silhueta que dá referência geográfica ao mapa."""
    if boundary is None or not len(boundary):
        return
    boundary.to_crs(crs).boundary.plot(
        ax=ax, color="black", linewidth=1.2, linestyle="--", alpha=0.7, zorder=5
    )


def _choropleth(vor_gdf, gdf, column, title, cmap, exporter, name, config,
                categorical=False, vcenter=None, boundary=None, log=False):
    """Mapa temático das regiões de influência das antenas.

    ``log`` serve às grandezas de cauda longa (densidade, volume): numa escala linear, um
    punhado de regiões centrais satura o topo e o resto da cidade vira um borrão de uma cor só.
    """
    fig, ax = plt.subplots(figsize=(11, 11))
    kwargs = dict(column=column, cmap=cmap, legend=True, edgecolor="black",
                  linewidth=0.4, alpha=0.65, ax=ax,
                  legend_kwds={"shrink": 0.55} if not categorical else None)
    if categorical:
        kwargs["categorical"] = True
        kwargs.pop("legend_kwds")
    elif log:
        values = vor_gdf[column].replace(0, np.nan).dropna()
        if len(values):
            kwargs["norm"] = LogNorm(vmin=values.min(), vmax=values.max())
    elif vcenter is not None:
        values = vor_gdf[column].dropna()
        span = max(abs(values.min() - vcenter), abs(values.max() - vcenter)) if len(values) else 1
        kwargs.update(vmin=vcenter - span, vmax=vcenter + span)
    vor_gdf.plot(**kwargs)
    gdf.plot(ax=ax, color="black", markersize=5)
    _plot_boundary(ax, boundary, gdf.crs)
    _add_basemap(ax, config)
    ax.set_axis_off()
    ax.set_title(title)
    exporter.save_figure(fig, name, "spatial")


# --------------------------------------------------------------------------- fluxos
def _plot_flow_map(flows, gdf, config, exporter, title, name, value="q_calls",
                   color="crimson", boundary=None):
    """Desenha os fluxos como linhas entre antenas, com largura proporcional ao volume."""
    xy = gdf.set_index("antenna_id").geometry
    sel = flows[flows["a"].isin(xy.index) & flows["b"].isin(xy.index)].copy()
    if sel.empty:
        return
    sel["geometry"] = [LineString([xy[a], xy[b]]) for a, b in zip(sel["a"], sel["b"])]
    sel_gdf = gpd.GeoDataFrame(sel, geometry="geometry", crs=gdf.crs)
    lw = 0.2 + 4.0 * sel_gdf[value] / sel_gdf[value].max()

    fig, ax = plt.subplots(figsize=(11, 11))
    sel_gdf.plot(ax=ax, color=color, linewidth=lw, alpha=0.45)
    gdf.plot(ax=ax, color="black", markersize=12)
    _plot_boundary(ax, boundary, gdf.crs)
    _add_basemap(ax, config)
    ax.set_axis_off()
    ax.set_title(title)
    exporter.save_figure(fig, name, "spatial")


def _gravity(flows, gdf, config, exporter, city_name, boundary=None) -> dict:
    """Modelo de gravidade e o mapa dos pares que fogem dele."""
    fit = fit_gravity_model(flows)
    if fit is None:
        logger.warning("[spatial] fluxos insuficientes — modelo de gravidade pulado")
        return {}

    df = fit["flows"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(df["dist_km"], df["q_calls"], s=8, alpha=0.25, color="steelblue")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("distância entre regiões (km)")
    axes[0].set_ylabel("chamadas no fluxo")
    axes[0].set_title(f"Decaimento com a distância (expoente b = {fit['distance_exponent']:.2f})")
    axes[1].scatter(df["log_predicted"], df["log_observed"], s=8, alpha=0.25, color="darkorange")
    lims = [df["log_predicted"].min(), df["log_predicted"].max()]
    axes[1].plot(lims, lims, "--", color="black", lw=1)
    axes[1].set_xlabel("log do fluxo previsto pela gravidade")
    axes[1].set_ylabel("log do fluxo observado")
    axes[1].set_title(f"Ajuste do modelo (R² = {fit['r2']:.2f})")
    fig.tight_layout()
    exporter.save_figure(fig, "gravity_model", "spatial")

    # os pares que mais superam a previsão: afinidades que tamanho e distância não explicam
    top = df.nlargest(min(150, len(df)), "residual")
    _plot_flow_map(
        top, gdf, config, exporter,
        f"Ligações acima do previsto pela gravidade — {city_name}",
        "gravity_residuals", value="residual", color="darkviolet", boundary=boundary,
    )
    exporter.save_data(
        df.nlargest(min(50, len(df)), "residual")[
            ["a", "b", "q_calls", "dist_km", "n_pairs", "residual"]
        ].round(3),
        "gravity_top_residuals.csv",
    )

    return {
        "gravity_distance_exponent": fit["distance_exponent"],
        "gravity_size_exponent": fit["size_exponent"],
        "gravity_r2": fit["r2"],
    }


# --------------------------------------------------------------------------- homofilia
def _homophily(nodes, flows, exporter, city_name, quintile_col="residence_quintile_state") -> dict:
    """Homofilia socioeconômica **ponderada pelo volume de chamadas** entre regiões.

    Diferente da versão por usuário, aqui o modelo nulo já embute a estrutura territorial:
    o que sobra é a preferência socioeconômica de fato, depois de descontado o efeito de
    "vizinho fala com vizinho". A matriz de mistura e o índice por quintil mostram que a
    média global esconde comportamentos opostos entre os estratos.
    """
    q = nodes.set_index("antenna_id")[quintile_col]
    qa = q.reindex(flows["a"]).to_numpy()
    qb = q.reindex(flows["b"]).to_numpy()
    w = flows["q_calls"].to_numpy(dtype=float)

    ok = pd.notna(qa) & pd.notna(qb) & np.isfinite(w)
    qa, qb, w = qa[ok], qb[ok], w[ok]
    if len(w) == 0 or len(set(qa)) < 2:
        logger.warning("[spatial] homofilia pulada (quintis insuficientes)")
        return {}

    total = w.sum()
    observed = float(w[qa == qb].sum() / total)

    # modelo nulo: embaralha o quintil entre as regiões, mantendo os fluxos no lugar
    index = pd.Index(nodes["antenna_id"])
    ia, ib = index.get_indexer(flows["a"][ok]), index.get_indexer(flows["b"][ok])
    codes = q.reindex(nodes["antenna_id"]).to_numpy()
    rng = np.random.default_rng(42)
    null = [
        float(w[perm[ia] == perm[ib]].sum() / total)
        for perm in (rng.permutation(codes) for _ in range(200))
    ]
    null_mean = float(np.mean(null))
    ratio = observed / null_mean if null_mean else float("nan")

    # matriz de mistura em "meias-arestas": e[i][j] = fração do volume ligando i a j
    mix = pd.DataFrame(0.0, index=QUINTIS, columns=QUINTIS)
    for source, target, value in zip(qa, qb, w):
        if source in mix.index and target in mix.columns:
            mix.loc[source, target] += value
            mix.loc[target, source] += value
    e = mix / mix.to_numpy().sum()
    a = e.sum(axis=1)

    # assortatividade de Newman ponderada por volume
    sum_e = float(np.trace(e.to_numpy()))
    sum_a2 = float((a**2).sum())
    assortativity = (sum_e - sum_a2) / (1 - sum_a2) if sum_a2 < 1 else float("nan")

    # índice de auto-preferência: observado / esperado se o volume fosse repartido ao acaso
    self_pref = pd.Series(
        {qi: (e.loc[qi, qi] / (a[qi] ** 2) if a[qi] > 0 else np.nan) for qi in QUINTIS}
    )

    row_norm = e.div(a.replace(0, np.nan), axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(row_norm, annot=True, fmt=".2f", cmap="rocket_r", ax=axes[0])
    axes[0].set_title("Para onde vai o volume de cada quintil")
    axes[0].set_xlabel("quintil da região do outro lado")
    axes[0].set_ylabel("quintil da região de origem")
    colors = ["crimson" if v > 1 else "steelblue" for v in self_pref.fillna(0)]
    axes[1].bar(self_pref.index, self_pref.values, color=colors)
    axes[1].axhline(1.0, color="black", ls="--", lw=1)
    axes[1].set_xlabel("quintil (q1 = 20% mais pobres, q5 = 20% mais ricos)")
    axes[1].set_ylabel("volume interno observado / esperado")
    axes[1].set_title("Índice de fechamento de cada estrato")
    fig.tight_layout()
    exporter.save_figure(fig, "homophily", "spatial")

    return {
        "homophily_observed": observed,
        "homophily_null": null_mean,
        "homophily_ratio": float(ratio),
        "homophily_assortativity_weighted": float(assortativity),
        "self_preference_index": {k: round(float(v), 3) for k, v in self_pref.items() if pd.notna(v)},
    }


# --------------------------------------------------------------------------- território
def _boundary_metrics(vor_gdf, boundary, gdf) -> dict:
    """Quanto território cada região cobre — e a densidade de moradores que isso implica.

    Só faz sentido com o Voronoi recortado por um limite real: sem ele, a área de uma célula
    de borda é um artefato do retângulo de corte. A densidade separa duas coisas que o
    ``n_users`` sozinho confunde — uma região com muita gente porque é grande e uma região
    com muita gente porque é adensada.
    """
    if vor_gdf is None or "area_km2" not in vor_gdf.columns:
        return {}

    area = vor_gdf["area_km2"]
    metrics = {
        "boundary_source": "GHS-FUA (GHSL/OECD R2019A)" if boundary is not None else "bbox",
        "region_area_km2_median": round(float(area.median()), 2),
        "region_area_km2_min": round(float(area.min()), 2),
        "region_area_km2_max": round(float(area.max()), 2),
    }

    if boundary is not None and len(boundary):
        metrics["city_area_km2"] = round(
            float(boundary.to_crs(_equal_area_crs(gdf)).union_all().area / 1e6), 1
        )

    if "n_users" in vor_gdf.columns:
        dens = np.where(area > 0, vor_gdf["n_users"] / area, np.nan)
        vor_gdf["users_per_km2"] = dens
        finite = pd.Series(dens).dropna()
        if len(finite):
            metrics["users_per_km2_median"] = round(float(finite.median()), 1)
            metrics["users_per_km2_max"] = round(float(finite.max()), 1)

    return metrics


# --------------------------------------------------------------------------- run
def run(net, config: dict, exporter, communities=None, nodes: pd.DataFrame | None = None) -> dict:
    """Calcula e exporta os mapas, a gravidade e a homofilia da rede de regiões."""
    logger.info("[spatial] Voronoi, corredores, gravidade, homofilia, insularidade")
    city_name = exporter.city_name
    nodes = net.nodes if nodes is None else nodes
    flows = net.flows

    gdf = _build_gdf(nodes)
    if gdf.empty:
        logger.warning("[spatial] antenas sem coordenadas — análise espacial pulada")
        return {"metrics": {}}

    # O limite territorial (GHS-FUA) recorta as células e vira o contorno de todos os mapas.
    boundary = load_boundary(config, crs=gdf.crs)

    vor_gdf = None
    if config.get("spatial", {}).get("voronoi", True):
        if len(gdf) >= 4:
            try:
                vor_gdf = _build_voronoi(gdf, boundary)
            except Exception as exc:
                logger.warning("[spatial] Voronoi não construído (%s)", exc)
        else:
            logger.warning("[spatial] poucas antenas (%d) — Voronoi pulado", len(gdf))

    metrics: dict = {"n_regions_mapped": int(len(gdf))}
    metrics.update(_boundary_metrics(vor_gdf, boundary, gdf))

    # -------- mapas temáticos das regiões --------
    if vor_gdf is not None:
        _choropleth(vor_gdf, gdf, "residence_quintile_state",
                    f"Regiões por quintil socioeconômico — {city_name}",
                    "RdYlGn", exporter, "voronoi_quintile", config, categorical=True,
                    boundary=boundary)
        _choropleth(vor_gdf, gdf, "insularity",
                    f"Insularidade: fração das chamadas que não sai da região — {city_name}",
                    "magma_r", exporter, "insularity_map", config, boundary=boundary)
        _choropleth(vor_gdf, gdf, "net_balance",
                    f"Balanço emissor (vermelho) × receptor (azul) — {city_name}",
                    "coolwarm", exporter, "net_balance_map", config, vcenter=0.0,
                    boundary=boundary)
        _choropleth(vor_gdf, gdf, "calls_per_user",
                    f"Chamadas por morador — {city_name}",
                    "viridis", exporter, "calls_per_user_map", config, boundary=boundary)
        if "users_per_km2" in vor_gdf.columns:
            _choropleth(vor_gdf, gdf, "users_per_km2",
                        f"Densidade: usuários da amostra por km² da região (escala log) — {city_name}",
                        "YlOrRd", exporter, "user_density_map", config, boundary=boundary, log=True)
        if "macro_region" in vor_gdf.columns and vor_gdf["macro_region"].notna().any():
            _choropleth(vor_gdf, gdf, "macro_region",
                        f"Macro-regiões funcionais detectadas nos fluxos — {city_name}",
                        "tab10", exporter, "macro_regions_map", config, categorical=True,
                        boundary=boundary)

    # -------- corredores --------
    _plot_flow_map(flows, gdf, config, exporter,
                   f"Todos os fluxos entre regiões — {city_name}", "flows_all",
                   boundary=boundary)
    backbone_flows = pd.DataFrame(
        [(u, v, d["q_calls"]) for u, v, d in net.backbone.edges(data=True)],
        columns=["a", "b", "q_calls"],
    )
    _plot_flow_map(backbone_flows, gdf, config, exporter,
                   f"Backbone: os corredores estruturantes — {city_name}",
                   "flows_backbone", color="darkred", boundary=boundary)

    # -------- gravidade e decaimento --------
    metrics.update(_gravity(flows, gdf, config, exporter, city_name, boundary))

    # -------- homofilia socioeconômica --------
    metrics.update(_homophily(nodes, flows, exporter, city_name))

    exporter.add_metrics("spatial", metrics)

    ratio = metrics.get("homophily_ratio")
    self_pref = metrics.get("self_preference_index", {})
    if ratio is not None:
        extremos = ""
        if "q5" in self_pref and "q1" in self_pref:
            extremos = (
                f" A média global esconde estratos que se comportam de forma oposta: o **q5 fala "
                f"consigo mesmo {self_pref['q5']:.2f}× mais do que o esperado**, enquanto o **q1 fica "
                f"em {self_pref['q1']:.2f}×** — as regiões ricas se fecham, as pobres se dispersam "
                f"pelos demais estratos."
            )
        homo_txt = (
            f"Entre regiões, **{metrics['homophily_observed']:.0%}** do volume liga áreas do mesmo "
            f"quintil, contra **{metrics['homophily_null']:.0%}** esperado ao acaso (**{ratio:.2f}×**)."
            + extremos
        )
    else:
        homo_txt = "A homofilia por quintil não pôde ser estimada (quintis insuficientes)."

    gravity_txt = ""
    if "gravity_distance_exponent" in metrics:
        gravity_txt = (
            f" O fluxo entre duas regiões segue um **modelo de gravidade**: cresce com o tamanho "
            f"delas (expoente {metrics['gravity_size_exponent']:.2f}) e cai com a distância "
            f"(**expoente {metrics['gravity_distance_exponent']:.2f}**, praticamente 1/d), "
            f"explicando {metrics['gravity_r2']:.0%} da variação. O que sobra do ajuste são "
            f"afinidades entre bairros que a geografia não explica."
        )

    territorio_txt = ""
    if "city_area_km2" in metrics:
        territorio_txt = (
            f" O recorte é a área urbana **funcional** da cidade (GHS-FUA), "
            f"**{metrics['city_area_km2']:,.0f} km²** — e não a divisa administrativa: as antenas da "
            f"base caem todas dentro dela. A célula mediana tem "
            f"**{metrics['region_area_km2_median']:.1f} km²**."
        )

    exporter.add_report_section(
        "Espaço, gravidade e segregação",
        f"Cada uma das **{len(gdf)} regiões** ocupa uma célula de Voronoi do mapa.{territorio_txt}"
        f"{gravity_txt} {homo_txt}",
    )

    return {"metrics": metrics}

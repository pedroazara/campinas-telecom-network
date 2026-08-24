"""Exporta as macro-regiões funcionais como um arquivo de polígonos (GeoJSON ou GeoPackage).

Cada macro-região é a **união das células de Voronoi** das antenas que o Louvain agrupou. O
resultado é um polígono por macro-região, com os atributos agregados — pronto para abrir no
QGIS, sobrepor às divisões administrativas oficiais ou entrar num slide.

O número de **partes desconexas** de cada polígono é o teste quantitativo do achado central do
projeto: o Louvain enxerga só volume de chamadas, nunca coordenadas. Se as macro-regiões saem
com uma parte só, a divisão funcional da cidade coincide com a territorial sem que ninguém
tenha dito ao algoritmo onde ficam as antenas.

Exemplos:
    python scripts/exportar_macroregioes.py campinas
    python scripts/exportar_macroregioes.py campinas --peso J
    python scripts/exportar_macroregioes.py campinas --formato gpkg --saida mapas/macro.gpkg

A insularidade da macro-região responde "quanto ela se basta": a fração do volume que nasce e
morre dentro dela, contando tanto as chamadas internas a cada antena quanto os fluxos entre
antenas do mesmo grupo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from networkx.algorithms.community import louvain_communities, modularity  # noqa: E402

from config import load_config  # noqa: E402
from src import antenna as antenna_module  # noqa: E402
from src.boundary import load_boundary  # noqa: E402
from src.pipeline.spatial import _build_gdf, _build_voronoi  # noqa: E402

FORMATOS = {"geojson": "GeoJSON", "gpkg": "GPKG", "shp": "ESRI Shapefile"}


def insularidade_por_macro(nodes: pd.DataFrame, flows: pd.DataFrame) -> pd.DataFrame:
    """Volume interno, externo e insularidade de cada macro-região.

    Interno = chamadas que não saem da própria antena + fluxos entre antenas do mesmo grupo.
    Externo = fluxos que cruzam a fronteira da macro-região.
    """
    grupo = nodes.set_index("antenna_id")["macro_region"]

    interno = nodes.groupby("macro_region")["calls_internal"].sum()

    f = flows.copy()
    f["ga"] = f["a"].map(grupo)
    f["gb"] = f["b"].map(grupo)
    f = f.dropna(subset=["ga", "gb"])

    mesmo = f[f["ga"] == f["gb"]].groupby("ga")["q_calls"].sum()
    interno = interno.add(mesmo, fill_value=0)

    cruza = f[f["ga"] != f["gb"]]
    externo = (
        pd.concat([
            cruza.groupby("ga")["q_calls"].sum(),
            cruza.groupby("gb")["q_calls"].sum(),
        ])
        .groupby(level=0)
        .sum()
    )

    out = pd.DataFrame({"calls_internal_macro": interno, "calls_external_macro": externo}).fillna(0)
    out["calls_total_macro"] = out["calls_internal_macro"] + out["calls_external_macro"]
    out["insularity_macro"] = (
        out["calls_internal_macro"] / out["calls_total_macro"].where(out["calls_total_macro"] > 0)
    )
    out.index.name = "macro_region"
    return out


def quintil_predominante(celulas: pd.DataFrame, macro: int, coluna: str) -> str | None:
    """Quintil da macro-região, ponderado pelos moradores (não pelo número de antenas)."""
    g = celulas[celulas["macro_region"] == macro]
    if coluna not in g.columns:
        return None
    peso = g.groupby(coluna, observed=True)["n_users"].sum()
    return str(peso.idxmax()) if len(peso) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cidade", help="identificador da cidade (config/<cidade>.yaml)")
    parser.add_argument("--peso", choices=["q_calls", "J", "n_pairs"],
                        help="peso das arestas no Louvain (padrão: o do config)")
    parser.add_argument("--seed", type=int, default=42, help="semente do Louvain (padrão: 42)")
    parser.add_argument("--formato", choices=list(FORMATOS), default="geojson")
    parser.add_argument("--saida", help="caminho do arquivo de saída")
    parser.add_argument("--sem-limite", action="store_true",
                        help="não recorta pelo limite da cidade (usa o retângulo antigo)")
    args = parser.parse_args()

    config = load_config(args.cidade)
    if args.peso:
        config.setdefault("antenna", {})["weight"] = args.peso

    data = config["data"]
    ea, an = ROOT / data["edges_antenna_path"], ROOT / data["antennas_path"]
    if not (ea.exists() and an.exists()):
        print(f"erro: parquets por antena não encontrados ({ea.name}, {an.name}).", file=sys.stderr)
        print(f"dica: rode antes `python main.py --city {args.cidade} --analyses eda`.",
              file=sys.stderr)
        return 1

    net = antenna_module.build(pd.read_parquet(ea), pd.read_parquet(an), config)

    comunidades = louvain_communities(net.G, weight="weight", seed=args.seed)
    Q = modularity(net.G, comunidades, weight="weight")
    de_quem = {a: i for i, c in enumerate(comunidades) for a in c}

    nodes = net.nodes.copy()
    nodes["macro_region"] = nodes["antenna_id"].map(de_quem)
    if nodes["macro_region"].isna().any():
        n = int(nodes["macro_region"].isna().sum())
        print(f"aviso: {n} antena(s) fora de qualquer comunidade — excluída(s).", file=sys.stderr)
        nodes = nodes.dropna(subset=["macro_region"])
    nodes["macro_region"] = nodes["macro_region"].astype(int)

    gdf = _build_gdf(nodes)
    limite = None if args.sem_limite else load_boundary(config, gdf.crs)
    if limite is None and not args.sem_limite:
        print("aviso: sem limite da cidade — as células de borda ficam com o corte retangular.",
              file=sys.stderr)
    celulas = _build_voronoi(gdf, limite)

    # A macro-região é a união das células das suas antenas.
    macro = celulas.dissolve(by="macro_region", aggfunc={"n_users": "sum"}).reset_index()
    macro["geometry"] = macro.geometry.buffer(0)   # remove slivers da união

    extra = (
        celulas.groupby("macro_region")
        .agg(n_antenas=("antenna_id", "size"), area_km2=("area_km2", "sum"))
    )
    macro = (
        macro.merge(extra, on="macro_region")
        .merge(insularidade_por_macro(nodes, net.flows), on="macro_region", how="left")
    )

    qcol = config.get("antenna", {}).get("quintile_column", "residence_quintile_state")
    macro["quintil_predominante"] = [
        quintil_predominante(celulas, m, qcol) for m in macro["macro_region"]
    ]
    macro["users_per_km2"] = macro["n_users"] / macro["area_km2"]
    macro["n_partes"] = macro.geometry.apply(
        lambda g: len(g.geoms) if g.geom_type == "MultiPolygon" else 1
    )
    macro["antenas"] = [
        ",".join(map(str, sorted(celulas.loc[celulas["macro_region"] == m, "antenna_id"])))
        for m in macro["macro_region"]
    ]

    for c in ("area_km2", "users_per_km2", "insularity_macro"):
        macro[c] = macro[c].round(3)
    macro = macro.sort_values("n_users", ascending=False).to_crs(epsg=4326)

    saida = ROOT / (args.saida or f"output/{config['city']}/data/macroregioes.{args.formato}")
    saida.parent.mkdir(parents=True, exist_ok=True)
    macro.to_file(saida, driver=FORMATOS[args.formato])

    peso = config.get("antenna", {}).get("weight", "q_calls")
    print(f"{len(macro)} macro-regiões (Louvain, peso={peso}, seed={args.seed}, Q={Q:.3f})\n")
    cols = ["macro_region", "n_antenas", "n_users", "area_km2", "insularity_macro",
            "quintil_predominante", "n_partes"]
    print(macro[cols].to_string(index=False))

    partidas = macro[macro["n_partes"] > 1]
    print()
    if partidas.empty:
        print("Todas as macro-regiões saíram espacialmente contíguas (uma parte cada) — e o")
        print("Louvain nunca viu uma coordenada sequer.")
    else:
        print(f"{len(partidas)} macro-região(ões) com partes desconexas: "
              f"{dict(zip(partidas['macro_region'], partidas['n_partes']))}")
    print(f"\nsalvo em {saida.relative_to(ROOT)} ({saida.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Entrypoint CLI do pipeline de análise de redes telefônicas multi-cidade.

Exemplos:
    python main.py --city campinas --analyses all
    python main.py --city all                       # roda todas as cidades configuradas
    python main.py --city campinas --analyses topology spatial
    python main.py --city campinas --no-basemap      # modo offline (sem tiles)
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # garante backend não-interativo antes de qualquer pyplot

from pathlib import Path

import pandas as pd
from networkx.algorithms.community import louvain_communities

from src.utils import load_config, setup_logging, CONFIG_DIR
from src.exporter import Exporter
from src import graph_builder
from src.pipeline import eda as eda_module, topology, spatial, advanced

ANALYSES = ["eda", "topology", "spatial", "advanced"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de análise de redes telefônicas urbanas (multi-cidade).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--city", required=True,
                        help="nome da cidade (config/<cidade>.yaml) ou 'all' para todas")
    parser.add_argument("--analyses", nargs="+", default=["all"], choices=["all"] + ANALYSES,
                        help="quais análises rodar")
    parser.add_argument("--output", default=None, help="pasta de saída (sobrescreve a config)")
    parser.add_argument("--no-basemap", action="store_true", help="não baixar tiles (offline)")
    parser.add_argument("--config", default=None, help="yaml extra mesclado sobre o default")
    return parser.parse_args()


def available_cities() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml") if p.stem != "default")


def run_city(city: str, args: argparse.Namespace, log) -> None:
    config = load_config(city, args.config)
    if args.output:
        config.setdefault("output", {})["base_dir"] = args.output
    if args.no_basemap:
        config.setdefault("spatial", {})["download_basemap"] = False

    requested = set(ANALYSES) if "all" in args.analyses else set(args.analyses)
    log.info("=== %s | análises: %s ===", config["city_name"], ", ".join(sorted(requested)))

    exporter = Exporter(config["city"], config["output"]["base_dir"], config)

    data = config["data"]
    ea_path, an_path = Path(data["edges_antenna_path"]), Path(data["antennas_path"])
    if ea_path.exists() and an_path.exists():
        log.info("Carregando parquets existentes (%s)", ea_path.name)
        edges_antenna = pd.read_parquet(ea_path)
        antennas = pd.read_parquet(an_path)
    else:
        log.info("Parquets ausentes — executando EDA (cache em dados/).")
        result = eda_module.run(config)
        edges_antenna, antennas = result["edges_antenna"], result["antennas"]

    if config.get("output", {}).get("save_data", True):
        exporter.save_data(edges_antenna, "edges_antenna.parquet")
        exporter.save_data(antennas, "antennas.parquet")

    G = graph_builder.build_graph(edges_antenna, config)
    G_main = graph_builder.get_main_component(G)
    log.info("Grafo: %d nós / %d arestas | gigante: %d nós",
             G.number_of_nodes(), G.number_of_edges(), G_main.number_of_nodes())

    # Louvain é caro em cidades grandes; calcula uma única vez e compartilha.
    communities = None
    if requested & {"topology", "spatial"}:
        log.info("Detectando comunidades (Louvain)...")
        communities = louvain_communities(G_main, weight="weight", seed=42)
        log.info("%d comunidades detectadas", len(communities))

    if "topology" in requested:
        topology.run(G, G_main, config, exporter, communities=communities)
    if "spatial" in requested:
        spatial.run(G_main, antennas, edges_antenna, config, exporter, communities=communities)
    if "advanced" in requested:
        advanced.run(G_main, config, exporter)

    exporter.save_metrics()
    exporter.write_report()
    log.info("Concluído: %s", exporter.base)


def main() -> None:
    args = parse_args()
    log = setup_logging()

    if args.city.lower() == "all":
        cities = available_cities()
        log.info("Rodando todas as cidades: %s", ", ".join(cities))
        for city in cities:
            try:
                run_city(city, args, log)
            except Exception:
                log.exception("Falha ao processar '%s' — seguindo para a próxima cidade", city)
        log.info("Todas as cidades processadas.")
    else:
        run_city(args.city, args, log)


if __name__ == "__main__":
    main()

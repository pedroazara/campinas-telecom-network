"""Entrypoint CLI do pipeline de análise de redes telefônicas multi-cidade.

Exemplos:
    python main.py --city campinas --analyses all
    python main.py --city campinas --analyses topology spatial
    python main.py --city campinas --analyses advanced --output ./resultados
    python main.py --city campinas --no-basemap
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")  # garante backend não-interativo antes de qualquer pyplot

from pathlib import Path

import pandas as pd

from src.utils import load_config, setup_logging
from src.exporter import Exporter
from src import graph_builder
from src.pipeline import eda as eda_module, topology, spatial, advanced

ANALYSES = ["eda", "topology", "spatial", "advanced"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline de análise de redes telefônicas urbanas (multi-cidade).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--city", required=True, help="nome da cidade (precisa de config/<cidade>.yaml)")
    parser.add_argument(
        "--analyses",
        nargs="+",
        default=["all"],
        choices=["all"] + ANALYSES,
        help="quais análises rodar",
    )
    parser.add_argument("--output", default=None, help="pasta de saída (sobrescreve a config)")
    parser.add_argument("--no-basemap", action="store_true", help="não baixar tiles do contextily (offline)")
    parser.add_argument("--config", default=None, help="yaml extra mesclado sobre o default")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log = setup_logging()

    config = load_config(args.city, args.config)
    if args.output:
        config.setdefault("output", {})["base_dir"] = args.output
    if args.no_basemap:
        config.setdefault("spatial", {})["download_basemap"] = False

    requested = set(ANALYSES) if "all" in args.analyses else set(args.analyses)
    log.info("Cidade: %s | análises: %s", config["city_name"], ", ".join(sorted(requested)))

    exporter = Exporter(config["city"], config["output"]["base_dir"], config)

    # ---------------- dados: usa parquets existentes ou roda a EDA ----------------
    data = config["data"]
    ea_path, an_path = Path(data["edges_antenna_path"]), Path(data["antennas_path"])

    if ea_path.exists() and an_path.exists():
        log.info("Carregando parquets existentes (%s, %s)", ea_path.name, an_path.name)
        edges_antenna = pd.read_parquet(ea_path)
        antennas = pd.read_parquet(an_path)
        if "eda" in requested:
            log.info("Parquets já existem — pulando a geração na EDA.")
    else:
        log.info("Parquets ausentes — executando EDA para gerá-los (cache em dados/).")
        result = eda_module.run(config)
        edges_antenna, antennas = result["edges_antenna"], result["antennas"]

    if config.get("output", {}).get("save_data", True):
        exporter.save_data(edges_antenna, "edges_antenna.parquet")
        exporter.save_data(antennas, "antennas.parquet")

    # ---------------- construção do grafo ----------------
    G = graph_builder.build_graph(edges_antenna, config)
    G_main = graph_builder.get_main_component(G)
    log.info(
        "Grafo: %d nós / %d arestas | componente gigante: %d nós",
        G.number_of_nodes(),
        G.number_of_edges(),
        G_main.number_of_nodes(),
    )

    # ---------------- análises ----------------
    if "topology" in requested:
        topology.run(G, G_main, config, exporter)
    if "spatial" in requested:
        spatial.run(G_main, antennas, edges_antenna, config, exporter)
    if "advanced" in requested:
        advanced.run(G_main, config, exporter)

    # ---------------- saída ----------------
    exporter.save_metrics()
    exporter.write_report()
    log.info("Concluído. Resultados em: %s", exporter.base)


if __name__ == "__main__":
    main()

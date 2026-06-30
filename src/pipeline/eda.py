"""EDA e construção das tabelas por antena (lógica do notebook 1).

Só é necessário quando `edges_antenna.parquet` / `antennas.parquet` ainda não existem,
pois depende do `residencias.csv` (~1 GB).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger("pipeline")


def run(config: dict) -> dict:
    """Gera os parquets por antena da cidade nos caminhos definidos na config.

    Escreve em `data.edges_antenna_path` / `data.antennas_path` (cache por cidade),
    de modo que execuções seguintes do pipeline pulem a leitura do `residencias.csv`.
    """
    data = config["data"]
    city_name = config.get("city_name", config["city"].capitalize())
    ea_path = Path(data["edges_antenna_path"])
    an_path = Path(data["antennas_path"])
    ea_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("[eda] lendo base agregada: %s", data["parquet_path"])
    base = pq.ParquetFile(data["parquet_path"]).read().to_pandas()

    logger.info("[eda] expandindo listas em arestas emissor→receptor")
    records = []
    for row in base.itertuples(index=False):
        for receiver, q_calls, distance_km, duration_total in zip(
            row.IDs_receivers_corr,
            row.q_calls_corr,
            row.residence_distance_km_corr,
            row.calls_duration_total_corr,
        ):
            records.append(
                {
                    "id_emisor": row.id_emisor,
                    "id_receiver": receiver,
                    "q_calls": q_calls,
                    "residence_distance_km": distance_km,
                    "calls_duration_total": duration_total,
                }
            )
    edges = pd.DataFrame(records)
    edges["avg_duration_per_call"] = edges["calls_duration_total"] / edges["q_calls"].replace(0, np.nan)

    logger.info("[eda] cruzando com residencias.csv (pode demorar — arquivo grande)")
    residencias = pd.read_csv(data["residencias_path"])
    res = residencias[
        [
            "ID",
            "residence_geometry",
            "residence_city",
            "residence_quintile_state",
            "residence_quintile_nation",
        ]
    ].copy()

    edges["id_emisor"] = edges["id_emisor"].astype(str).str.strip()
    edges["id_receiver"] = edges["id_receiver"].astype(str).str.strip()
    res["ID"] = res["ID"].astype(str).str.strip()
    ids_res = set(res["ID"])

    edges_antenna = edges[
        edges["id_emisor"].isin(ids_res) & edges["id_receiver"].isin(ids_res)
    ].copy()

    # cada geometria residencial distinta = uma antena
    antenna_key = res["residence_geometry"].astype("string").fillna("<NA>")
    res["antenna_id"] = pd.factorize(antenna_key, sort=True)[0] + 1

    antennas = (
        res[
            [
                "antenna_id",
                "residence_geometry",
                "residence_city",
                "residence_quintile_state",
                "residence_quintile_nation",
            ]
        ]
        .drop_duplicates(subset=["antenna_id"])
        .sort_values("antenna_id")
        .reset_index(drop=True)
    )
    antennas = antennas[antennas["residence_city"] == city_name]

    user_to_antenna = res.set_index("ID")["antenna_id"]
    edges_antenna["emissor_antenna_id"] = edges_antenna["id_emisor"].map(user_to_antenna).astype("Int64")
    edges_antenna["receptor_antenna_id"] = edges_antenna["id_receiver"].map(user_to_antenna).astype("Int64")

    edges_antenna.to_parquet(ea_path, index=False)
    antennas.to_parquet(an_path, index=False)
    logger.info(
        "[eda] gerados %d arestas e %d antenas (%s) em %s",
        len(edges_antenna),
        len(antennas),
        city_name,
        ea_path.parent,
    )

    return {
        "edges_antenna_path": ea_path,
        "antennas_path": an_path,
        "edges_antenna": edges_antenna,
        "antennas": antennas,
    }

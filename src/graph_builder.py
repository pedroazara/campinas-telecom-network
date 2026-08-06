"""Agregação no nível dos usuários — insumo para a construção da rede de regiões.

Estas funções são o passo intermediário entre a tabela bruta de chamadas e a rede de antenas
montada em :mod:`src.antenna`: elas juntam as chamadas A→B e B→A num único par e descobrem em
que antena cada usuário mora. O grafo de usuários em si não é mais construído — a unidade de
análise do projeto é a antena.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_edges_graph(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega arestas dirigidas A→B e B→A no mesmo par não-direcionado de usuários."""
    e = edges_df.copy()
    e["source"] = np.minimum(e["id_emisor"].astype(str), e["id_receiver"].astype(str))
    e["target"] = np.maximum(e["id_emisor"].astype(str), e["id_receiver"].astype(str))

    return (
        e.groupby(["source", "target"], as_index=False)
        .agg(
            q_calls=("q_calls", "sum"),
            calls_duration_total=("calls_duration_total", "sum"),
            avg_duration_per_call=("avg_duration_per_call", "mean"),
            residence_distance_km=("residence_distance_km", "mean"),
        )
    )


def build_user_antenna_map(edges_df: pd.DataFrame) -> pd.Series:
    """Mapeia cada usuário à sua antena residencial (moda das antenas observadas)."""
    ua = pd.concat(
        [
            edges_df[["id_emisor", "emissor_antenna_id"]].rename(
                columns={"id_emisor": "user_id", "emissor_antenna_id": "antenna_id"}
            ),
            edges_df[["id_receiver", "receptor_antenna_id"]].rename(
                columns={"id_receiver": "user_id", "receptor_antenna_id": "antenna_id"}
            ),
        ]
    )
    return ua.dropna().groupby("user_id")["antenna_id"].agg(lambda s: s.mode().iat[0])

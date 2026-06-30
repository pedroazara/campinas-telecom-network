"""Construção do grafo da rede de chamadas (lógica comum aos notebooks 2–4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx


def build_edges_graph(edges_df: pd.DataFrame) -> pd.DataFrame:
    """Agrega arestas dirigidas A→B e B→A no mesmo par não-direcionado.

    O peso é ``log1p(q_calls) * log1p(calls_duration_total)``.
    """
    e = edges_df.copy()
    e["source"] = np.minimum(e["id_emisor"].astype(str), e["id_receiver"].astype(str))
    e["target"] = np.maximum(e["id_emisor"].astype(str), e["id_receiver"].astype(str))

    edges_graph = (
        e.groupby(["source", "target"], as_index=False)
        .agg(
            q_calls=("q_calls", "sum"),
            calls_duration_total=("calls_duration_total", "sum"),
            avg_duration_per_call=("avg_duration_per_call", "mean"),
            residence_distance_km=("residence_distance_km", "mean"),
        )
    )
    edges_graph["weight"] = (
        np.log1p(edges_graph["q_calls"]) * np.log1p(edges_graph["calls_duration_total"])
    )
    return edges_graph


def build_graph(edges_df: pd.DataFrame, config: dict | None = None) -> nx.Graph:
    """Constrói o grafo não-direcionado ponderado a partir das arestas por antena."""
    edges_graph = build_edges_graph(edges_df)
    G = nx.from_pandas_edgelist(
        edges_graph,
        source="source",
        target="target",
        edge_attr=[
            "weight",
            "q_calls",
            "calls_duration_total",
            "avg_duration_per_call",
            "residence_distance_km",
        ],
    )
    if config and config.get("graph", {}).get("remove_self_loops"):
        # self-loops são tratados nas métricas estruturais; aqui mantemos o grafo bruto.
        pass
    return G


def get_main_component(G: nx.Graph) -> nx.Graph:
    """Retorna a componente gigante (maior componente conexa) como cópia independente."""
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def remove_self_loops(G: nx.Graph) -> nx.Graph:
    """Retorna uma cópia de G sem self-loops (autochamadas)."""
    H = G.copy()
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    return H


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

"""Construção da rede de antenas — a unidade de análise do projeto.

Cada antena (geometria residencial distinta) é um **nó**; as pessoas que moram sob ela
viram atributos agregados. As arestas são os **fluxos de chamadas entre regiões**.

Diferenças importantes em relação à rede de usuários:

- A rede resultante é pequena e densa (em Campinas, 145 nós e densidade ≈ 0,56). Métricas
  que dependem de cauda de grau (lei de potência, small-world contra Erdős–Rényi, k-core)
  perdem sentido; o que informa aqui são **pesos**, **espaço** e **backbone**.
- As chamadas dentro da mesma antena — mais de um terço do volume — deixam de ser arestas e
  viram um atributo do nó (`calls_internal`), a medida de insularidade da região.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import networkx as nx
from shapely import wkb

from .graph_builder import build_edges_graph, build_user_antenna_map

logger = logging.getLogger("pipeline")

EARTH_RADIUS_KM = 6371.0088


# --------------------------------------------------------------------------- geometria
def parse_point(value):
    """Lê a geometria residencial (WKB, possivelmente como repr de bytes) num ponto shapely."""
    if value is None or isinstance(value, float):
        return None
    if isinstance(value, str):
        try:
            return wkb.loads(ast.literal_eval(value))
        except (ValueError, SyntaxError):
            return wkb.loads(value, hex=True)
    if isinstance(value, (bytes, bytearray)):
        return wkb.loads(bytes(value))
    return value  # já é uma geometria


def haversine_km(lon1, lat1, lon2, lat2):
    """Distância em km sobre a esfera. Vetorizado; evita a distorção do EPSG:3857."""
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


# --------------------------------------------------------------------------- nós
def build_antenna_nodes(edges_antenna: pd.DataFrame, antennas: pd.DataFrame) -> pd.DataFrame:
    """Tabela de nós: uma linha por antena, com os moradores agregados.

    Colunas principais:
      - ``n_users``            usuários residentes distintos sob a antena
      - ``calls_out/in``       volume emitido e recebido (chamadas internas entram nos dois)
      - ``calls_internal``     chamadas entre dois moradores da própria antena
      - ``calls_total``        volume que toca a antena, contando as internas uma vez só
      - ``insularity``         fração do volume que não sai da região
      - ``net_balance``        (out − in) / (out + in): +1 só emite, −1 só recebe
    """
    ant = antennas.drop_duplicates(subset=["antenna_id"]).copy()

    geom = ant["residence_geometry"].apply(parse_point)
    ant["lon"] = geom.apply(lambda p: p.x if p is not None else np.nan)
    ant["lat"] = geom.apply(lambda p: p.y if p is not None else np.nan)
    ant = ant.drop(columns=["residence_geometry"])

    user_antenna = build_user_antenna_map(edges_antenna)
    n_users = user_antenna.value_counts().rename("n_users")

    e = edges_antenna.dropna(subset=["emissor_antenna_id", "receptor_antenna_id"])
    out = e.groupby("emissor_antenna_id").agg(
        calls_out=("q_calls", "sum"), duration_out=("calls_duration_total", "sum")
    )
    inn = e.groupby("receptor_antenna_id").agg(
        calls_in=("q_calls", "sum"), duration_in=("calls_duration_total", "sum")
    )
    internal = (
        e[e["emissor_antenna_id"] == e["receptor_antenna_id"]]
        .groupby("emissor_antenna_id")
        .agg(calls_internal=("q_calls", "sum"), duration_internal=("calls_duration_total", "sum"))
    )

    nodes = (
        ant.set_index("antenna_id")
        .join([n_users, out, inn, internal])
        .fillna({
            "n_users": 0, "calls_out": 0, "calls_in": 0, "calls_internal": 0,
            "duration_out": 0, "duration_in": 0, "duration_internal": 0,
        })
        .reset_index()
    )

    nodes["calls_total"] = nodes["calls_out"] + nodes["calls_in"] - nodes["calls_internal"]
    nodes["calls_external"] = nodes["calls_total"] - nodes["calls_internal"]
    nodes["insularity"] = np.where(
        nodes["calls_total"] > 0, nodes["calls_internal"] / nodes["calls_total"], np.nan
    )
    denom = nodes["calls_out"] + nodes["calls_in"]
    nodes["net_balance"] = np.where(denom > 0, (nodes["calls_out"] - nodes["calls_in"]) / denom, np.nan)
    nodes["calls_per_user"] = np.where(
        nodes["n_users"] > 0, nodes["calls_total"] / nodes["n_users"], np.nan
    )
    nodes["n_users"] = nodes["n_users"].astype(int)

    return nodes.sort_values("antenna_id").reset_index(drop=True)


# --------------------------------------------------------------------------- arestas
def build_antenna_flows(edges_antenna: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Fluxos não-direcionados entre antenas distintas (as internas ficam nos nós).

    ``n_pairs`` é o número de pares de pessoas por trás do fluxo: separa um corredor
    "largo" (muita gente conversando pouco) de um "estreito" (poucos laços intensos).
    ``intensity`` normaliza pelo tamanho das duas regiões — sem isso, o mapa de fluxos
    apenas redesenha onde mora mais gente.
    """
    user_antenna = build_user_antenna_map(edges_antenna)
    pairs = build_edges_graph(edges_antenna)  # pares de usuários não-direcionados

    pairs["ant_source"] = pairs["source"].map(user_antenna)
    pairs["ant_target"] = pairs["target"].map(user_antenna)
    pairs = pairs.dropna(subset=["ant_source", "ant_target"])

    valid = set(nodes["antenna_id"])
    pairs = pairs[pairs["ant_source"].isin(valid) & pairs["ant_target"].isin(valid)]

    pairs["a"] = np.minimum(pairs["ant_source"], pairs["ant_target"])
    pairs["b"] = np.maximum(pairs["ant_source"], pairs["ant_target"])
    pairs = pairs[pairs["a"] != pairs["b"]]

    flows = pairs.groupby(["a", "b"], as_index=False).agg(
        q_calls=("q_calls", "sum"),
        calls_duration_total=("calls_duration_total", "sum"),
        n_pairs=("q_calls", "size"),
        mean_user_distance_km=("residence_distance_km", "mean"),
    )

    coords = nodes.set_index("antenna_id")[["lon", "lat"]]
    flows["dist_km"] = haversine_km(
        coords["lon"].reindex(flows["a"]).to_numpy(),
        coords["lat"].reindex(flows["a"]).to_numpy(),
        coords["lon"].reindex(flows["b"]).to_numpy(),
        coords["lat"].reindex(flows["b"]).to_numpy(),
    )

    users = nodes.set_index("antenna_id")["n_users"]
    size_product = users.reindex(flows["a"]).to_numpy() * users.reindex(flows["b"]).to_numpy()
    flows["size_product"] = size_product
    flows["intensity"] = np.where(size_product > 0, flows["q_calls"] / size_product, np.nan)
    flows["weight"] = flows["q_calls"].astype(float)
    flows["avg_duration_per_call"] = flows["calls_duration_total"] / flows["q_calls"].replace(0, np.nan)

    return flows.sort_values("q_calls", ascending=False).reset_index(drop=True)


def build_directed_flows(edges_antenna: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Fluxos dirigidos entre antenas distintas — base da reciprocidade e do balanço líquido."""
    valid = set(nodes["antenna_id"])
    e = edges_antenna.dropna(subset=["emissor_antenna_id", "receptor_antenna_id"]).copy()
    e = e[e["emissor_antenna_id"].isin(valid) & e["receptor_antenna_id"].isin(valid)]
    e = e[e["emissor_antenna_id"] != e["receptor_antenna_id"]]
    return e.groupby(
        ["emissor_antenna_id", "receptor_antenna_id"], as_index=False
    ).agg(
        q_calls=("q_calls", "sum"),
        calls_duration_total=("calls_duration_total", "sum"),
        n_pairs=("q_calls", "size"),
    ).rename(columns={"emissor_antenna_id": "source", "receptor_antenna_id": "target"})


# --------------------------------------------------------------------------- grafos
def build_antenna_graph(flows: pd.DataFrame, nodes: pd.DataFrame) -> nx.Graph:
    """Grafo não-direcionado ponderado das regiões, com os atributos dos nós anexados."""
    attrs = [
        "n_users", "calls_total", "calls_internal", "insularity", "net_balance",
        "calls_per_user", "lon", "lat", "residence_quintile_state", "residence_quintile_nation",
    ]
    G = nx.from_pandas_edgelist(
        flows, "a", "b",
        edge_attr=["weight", "q_calls", "n_pairs", "dist_km", "intensity", "calls_duration_total"],
    )
    G.add_nodes_from(nodes["antenna_id"])  # antenas isoladas não podem sumir
    present = [c for c in attrs if c in nodes.columns]
    nx.set_node_attributes(G, nodes.set_index("antenna_id")[present].to_dict("index"))
    return G


def build_antenna_digraph(directed: pd.DataFrame, nodes: pd.DataFrame) -> nx.DiGraph:
    """Versão dirigida: quem liga para quem entre regiões."""
    D = nx.from_pandas_edgelist(
        directed, "source", "target", edge_attr=["q_calls", "n_pairs"], create_using=nx.DiGraph
    )
    D.add_nodes_from(nodes["antenna_id"])
    return D


# --------------------------------------------------------------------------- backbone
def disparity_filter(G: nx.Graph, alpha: float = 0.05, weight: str = "weight") -> nx.Graph:
    """Backbone por filtro de disparidade (Serrano, Boguñá & Vespignani, 2009).

    Numa rede quase completa, quase toda aresta existe — o que distingue um corredor real
    é ele carregar **mais peso do que o esperado** se a força do nó fosse repartida ao acaso
    entre seus vizinhos. Para cada extremo calcula-se

        α_ij = (1 − w_ij / s_i)^(k_i − 1)

    e a aresta entra no backbone se for significativa para pelo menos um dos dois lados.
    Isso preserva os fluxos importantes tanto de regiões grandes quanto pequenas — coisa que
    um corte por peso absoluto não faz.
    """
    strength = dict(G.degree(weight=weight))
    degree = dict(G.degree())

    keep = []
    for u, v, data in G.edges(data=True):
        w = data.get(weight, 1.0)
        alphas = []
        for node, other in ((u, v), (v, u)):
            k, s = degree[node], strength[node]
            if k > 1 and s > 0:
                alphas.append((1 - w / s) ** (k - 1))
        # nós de grau 1 não têm alternativa: a aresta é a única ligação e é preservada
        if not alphas or min(alphas) < alpha:
            keep.append((u, v))

    B = nx.Graph()
    B.add_nodes_from(G.nodes(data=True))
    B.add_edges_from((u, v, G.edges[u, v]) for u, v in keep)
    return B


def s_core_levels(G: nx.Graph, weight: str = "weight", n_levels: int = 60) -> pd.Series:
    """Decomposição s-core: k-core generalizado para pesos (Eidsaa & Almaas, 2013).

    O k-core comum é inútil aqui — numa rede densa quase todo nó tem grau alto. O s-core
    poda por **força** (volume de chamadas), revelando o núcleo de regiões que concentram o
    tráfego da cidade. Retorna, por nó, a maior força-limiar em que ele ainda sobrevive.
    """
    if G.number_of_edges() == 0:
        return pd.Series(0.0, index=list(G.nodes()))

    max_strength = max(dict(G.degree(weight=weight)).values())
    levels = pd.Series(0.0, index=list(G.nodes()), dtype=float)

    for threshold in np.linspace(0, max_strength, n_levels)[1:]:
        H = G.copy()
        while True:
            weak = [n for n, s in H.degree(weight=weight) if s < threshold]
            if not weak:
                break
            H.remove_nodes_from(weak)
        if H.number_of_nodes() == 0:
            break
        levels[list(H.nodes())] = threshold

    return levels


# --------------------------------------------------------------------------- gravidade
def fit_gravity_model(flows: pd.DataFrame) -> dict | None:
    """Ajusta o modelo de gravidade  F_ij ≈ C · (n_i n_j)^a / d_ij^b  por MQO em log-log.

    É a formalização do "decaimento com a distância": em vez de só mostrar a curva caindo,
    estima **quanto** ela cai (b) e quanto do fluxo o tamanho das regiões e a distância
    conseguem explicar (R²). O resíduo de cada par — o que sobra depois de descontar tamanho
    e distância — aponta as afinidades que a geografia não explica.
    """
    df = flows[(flows["q_calls"] > 0) & (flows["dist_km"] > 0) & (flows["size_product"] > 0)].copy()
    if len(df) < 20:
        return None

    y = np.log(df["q_calls"].to_numpy(dtype=float))
    X = np.column_stack([
        np.ones(len(df)),
        np.log(df["size_product"].to_numpy(dtype=float)),
        np.log(df["dist_km"].to_numpy(dtype=float)),
    ])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    df["log_observed"] = y
    df["log_predicted"] = pred
    df["residual"] = resid

    return {
        "const": float(coef[0]),
        "size_exponent": float(coef[1]),
        "distance_exponent": float(-coef[2]),  # b, positivo quando o fluxo cai com a distância
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "n_flows": int(len(df)),
        "flows": df,
    }


# --------------------------------------------------------------------------- eficiência
def weighted_global_efficiency(G: nx.Graph, weight: str = "weight") -> float:
    """Eficiência global usando 1/peso como custo de travessia.

    Numa rede densa a componente gigante nunca se fragmenta, então a robustez não pode ser
    medida por fragmentação. A eficiência ponderada degrada continuamente e capta o que
    interessa: o quanto a cidade perde capacidade de comunicação ao perder regiões.
    """
    n = G.number_of_nodes()
    if n < 2:
        return 0.0
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    H.add_edges_from(
        (u, v, {"cost": 1.0 / w if w > 0 else np.inf})
        for u, v, w in G.edges(data=weight, default=1.0)
    )
    total = 0.0
    for source, lengths in nx.all_pairs_dijkstra_path_length(H, weight="cost"):
        total += sum(1.0 / d for target, d in lengths.items() if target != source and d > 0)
    return total / (n * (n - 1))


# --------------------------------------------------------------------------- fachada
@dataclass
class AntennaNetwork:
    """Tudo o que as análises precisam da rede de regiões, construído uma única vez."""

    nodes: pd.DataFrame
    flows: pd.DataFrame
    directed: pd.DataFrame
    G: nx.Graph
    D: nx.DiGraph
    backbone: nx.Graph
    alpha: float

    @property
    def n_antennas(self) -> int:
        return len(self.nodes)

    @property
    def internal_call_share(self) -> float:
        """Fração de **todas** as chamadas da cidade que não saem da região de origem."""
        internal = float(self.nodes["calls_internal"].sum())
        external = float(self.flows["q_calls"].sum())
        total = internal + external
        return internal / total if total else float("nan")


def build(edges_antenna: pd.DataFrame, antennas: pd.DataFrame, config: dict | None = None) -> AntennaNetwork:
    """Constrói a rede de antenas completa a partir das tabelas por usuário."""
    alpha = (config or {}).get("antenna", {}).get("backbone_alpha", 0.05)

    nodes = build_antenna_nodes(edges_antenna, antennas)
    flows = build_antenna_flows(edges_antenna, nodes)
    directed = build_directed_flows(edges_antenna, nodes)
    G = build_antenna_graph(flows, nodes)
    D = build_antenna_digraph(directed, nodes)
    backbone = disparity_filter(G, alpha=alpha)

    logger.info(
        "[antena] %d regiões | %d fluxos (densidade %.3f) | backbone %d fluxos (α=%.2f)",
        len(nodes), G.number_of_edges(), nx.density(G), backbone.number_of_edges(), alpha,
    )
    return AntennaNetwork(nodes, flows, directed, G, D, backbone, alpha)

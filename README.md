# Caracterização de Redes Telefônicas Urbanas por Métodos de Redes Complexas

Pipeline modular e configurável para analisar a **rede telefônica de qualquer cidade** a partir de
dados anonimizados de chamadas cruzados com residência e quintis socioeconômicos. A análise
caracteriza a estrutura social e espacial da cidade (topologia, comunidades, segregação
socioeconômica, hubs, robustez) e exporta figuras, métricas e um relatório por cidade.

## Pipeline de produção

```bash
python main.py --city campinas --analyses all
python main.py --city campinas --analyses topology spatial
python main.py --city campinas --analyses advanced --output ./resultados
python main.py --city campinas --no-basemap          # modo offline (sem tiles)
python main.py --city lavras  --analyses all
```

Argumentos:

| Argumento | Descrição |
|---|---|
| `--city` | nome da cidade (precisa de `config/<cidade>.yaml`) |
| `--analyses` | `all` (padrão) ou qualquer combinação de `eda topology spatial advanced` |
| `--output` | pasta de saída (sobrescreve a config) |
| `--no-basemap` | não baixar tiles do `contextily` (roda sem internet) |
| `--config` | YAML extra mesclado por cima do `default.yaml` |

### Saída gerada (`output/<cidade>/`)

```
output/campinas/
├── data/      edges_antenna.parquet, antennas.parquet, top_hubs.csv, antenna_flows.csv
├── figures/
│   ├── topology/   degree_distribution.png, ccdf.png, communities.png
│   ├── spatial/    voronoi_map.png, homophily_matrix.png, distance_decay.png,
│   │               antenna_network.png, hubs_map.png
│   └── advanced/   powerlaw_fit.png, assortativity.png, kcore.png, smallworld.png, robustness.png
└── summary/   metrics.json   (todas as métricas)
            report.md      (relatório com tabela de métricas + tradução "para o prefeito")
```

### Adicionar uma nova cidade

Basta criar `config/<cidade>.yaml` apontando para os dados — **sem alterar código**:

```yaml
city: saopaulo
city_name: "Sao Paulo"        # valor exato em residence_city
data:
  parquet_path: "./dados/SaoPaulo.parquet"
  residencias_path: "./dados/residencias.csv"
  edges_antenna_path: "./dados/saopaulo_edges_antenna.parquet"
  antennas_path: "./dados/saopaulo_antennas.parquet"
```

Se os parquets por antena ainda não existirem, o módulo de EDA os gera a partir do
`residencias.csv` (e os armazena em `dados/` como cache para as próximas execuções).

### Configuração

- `config/default.yaml` — parâmetros padrão (grafo, análises, output, espacial, avançado).
- `config/<cidade>.yaml` — caminhos dos dados da cidade; sobrescreve o default.
- A config efetiva é `default.yaml` + `<cidade>.yaml` + (opcional) `--config extra.yaml`.

## Estrutura do projeto

```
├── src/
│   ├── graph_builder.py      # constrói o grafo (comum a todas as análises)
│   ├── exporter.py           # salva figuras, métricas (JSON) e relatório (Markdown)
│   ├── utils.py              # load_config, logging, criação de pastas
│   └── pipeline/
│       ├── eda.py            # gera os parquets por antena (a partir do residencias.csv)
│       ├── topology.py       # grau, componentes, clustering, centralidades, comunidades
│       ├── spatial.py        # Voronoi, homofilia socioeconômica, decaimento, rede de antenas
│       └── advanced.py       # scale-free, assortatividade/k-core, small-world, robustez
├── config/                   # default.yaml + um yaml por cidade
├── notebooks/                # notebooks exploratórios originais (referência)
├── dados/                    # parquets de entrada/cache (residencias.csv não versionado)
├── output/                   # gerado em runtime (não versionado)
├── main.py                   # entrypoint CLI
└── config.py                 # config dos notebooks + load_config das cidades
```

## Instalação

O projeto usa **[uv](https://docs.astral.sh/uv/)** (há um `.venv` com Python 3.13):

```bash
uv sync                 # instala as dependências do pyproject.toml
```

ou, com pip:

```bash
pip install -r requirements.txt
```

Dependências principais: `networkx`, `pandas`, `numpy`, `scipy`, `geopandas`, `shapely`,
`contextily` (baixa o mapa de fundo — exige internet), `seaborn`, `matplotlib`, `pyarrow`, `pyyaml`.

## Notebooks de referência (`notebooks/`)

Os notebooks exploratórios originais documentam, célula a célula, a mesma análise que o pipeline
automatiza. Eles importam a cidade de `config.py` (`from config import CITY_NAME`).

- `1-eda.ipynb` — EDA da base e construção das tabelas por antena.
- `2-rede-complexa.ipynb` — topologia: grau, CCDF, componentes, clustering, centralidades, comunidades.
- `3-analise-espacial.ipynb` — Voronoi, comunidades no mapa, decaimento, homofilia socioeconômica, rede de antenas.
- `4-analises-avancadas.ipynb` — lei de potência, assortatividade/k-core, small-world, robustez.

## Análises incluídas

**Topologia** — grafo não-direcionado ponderado, densidade, distribuição de grau e CCDF,
componentes e componente gigante, clustering, centralidades (grau, força, intermediação, autovetor)
e comunidades (Louvain + modularidade).

**Espacial e socioeconômica** — diagrama de Voronoi por quintil, **homofilia socioeconômica**
(observado vs. modelo nulo + matriz de mistura), **decaimento da intensidade com a distância**,
**rede agregada de fluxo entre antenas** e mapa de hubs.

**Avançadas** — **lei de potência** (MLE/Clauset), **assortatividade** de grau e **k-core**,
**small-world** (σ vs. grafo aleatório) e **robustez** (ataque dirigido vs. falha aleatória).

## Próximos passos sugeridos

- Comparar cidades (Campinas × Lavras × outras) usando os relatórios por cidade.
- Incorporar métricas temporais, caso exista base com timestamps das chamadas.
- Aprofundar a relação entre homofilia socioeconômica e segregação espacial.

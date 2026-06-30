# Instruções para o Agente de IA — Pipeline de Produção Multi-Cidade

> Este documento descreve o que o agente de IA deve construir na branch `feat/production-pipeline`.
> O objetivo é transformar os notebooks exploratórios em um pipeline modular, configurável e
> reutilizável para analisar redes telefônicas de **qualquer cidade**, exportando resultados
> organizados por cidade em uma pasta de output.

---

## 1. Visão geral do produto final

O usuário deve conseguir rodar:

```bash
python main.py --city campinas --analyses all
python main.py --city saopaulo --analyses topology spatial
python main.py --city campinas --analyses advanced --output ./resultados
```

E receber, ao final, uma pasta estruturada:

```
output/
└── campinas/
    ├── data/
    │   ├── edges_antenna.parquet
    │   └── antennas.parquet
    ├── figures/
    │   ├── topology/
    │   │   ├── degree_distribution.png
    │   │   ├── ccdf.png
    │   │   └── communities.png
    │   ├── spatial/
    │   │   ├── voronoi_map.png
    │   │   ├── homophily_matrix.png
    │   │   └── antenna_network.png
    │   └── advanced/
    │       ├── powerlaw_fit.png
    │       ├── robustness.png
    │       └── smallworld.png
    └── summary/
        ├── metrics.json
        └── report.md
```

---

## 2. Nova estrutura de pastas do projeto

O agente deve reorganizar o projeto assim:

```
projetos-dados-telefonicos/
├── src/
│   ├── __init__.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── eda.py            # lógica do notebook 1
│   │   ├── topology.py       # lógica do notebook 2
│   │   ├── spatial.py        # lógica do notebook 3
│   │   └── advanced.py       # lógica do notebook 4
│   ├── graph_builder.py      # constrói o grafo (lógica comum aos NBs 2–4)
│   ├── exporter.py           # salva figuras, JSONs, relatório markdown
│   └── utils.py              # helpers compartilhados
├── config/
│   ├── campinas.yaml         # exemplo de configuração de cidade
│   └── default.yaml          # valores padrão (pode ser sobrescrito por cidade)
├── dados/                    # mantém os parquets originais
├── notebooks/                # move os notebooks originais para cá (sem alterar)
│   ├── 1-eda.ipynb
│   ├── 2-rede-complexa.ipynb
│   ├── 3-analise-espacial.ipynb
│   └── 4-analises-avancadas.ipynb
├── output/                   # criado em runtime, não versionado
├── main.py                   # entrypoint CLI
├── config.py                 # leitura e merge de configs
├── requirements.txt
└── pyproject.toml
```

**Regras:**
- Os notebooks originais devem ser **movidos** para `notebooks/` sem modificação — são a referência.
- Todo código novo vai em `src/`.
- O `.gitignore` deve incluir `output/`.

---

## 3. Sistema de configuração (`config/`)

### `config/default.yaml`

```yaml
graph:
  weighted: true
  directed: false
  weight_formula: "log1p(q_calls) * log1p(total_duration)"
  remove_self_loops: true

analyses:
  eda: true
  topology: true
  spatial: true
  advanced: true

output:
  base_dir: "./output"
  save_figures: true
  save_data: true
  save_report: true
  figure_format: "png"
  figure_dpi: 150

spatial:
  download_basemap: true   # contextily — requer internet
  voronoi: true

advanced:
  robustness_steps: 50
  betweenness_sample_k: 500
  powerlaw_xmin_auto: true
```

### `config/campinas.yaml`

```yaml
city: campinas
data:
  parquet_path: "./dados/Campinas.parquet"
  residencias_path: "./dados/residencias.csv"
  edges_path: "./dados/edges.csv"
  edges_antenna_path: "./dados/edges_antenna.parquet"
  antennas_path: "./dados/antennas.parquet"
```

O agente deve criar a estrutura de config para que **adicionar uma nova cidade** seja apenas criar
um novo arquivo `config/<cidade>.yaml` apontando para os dados correspondentes.

---

## 4. Módulos a implementar

### 4.1 `src/graph_builder.py`

Extrair do setup comum dos notebooks 2–4:

- `build_graph(edges_df) -> nx.Graph`: constrói grafo não-direcionado ponderado com a fórmula de peso
- `get_main_component(G) -> nx.Graph`: retorna a componente gigante
- `remove_self_loops(G) -> nx.Graph`

### 4.2 `src/pipeline/eda.py`

Extrair do notebook 1:

- `run(config) -> dict`: lê `Campinas.parquet` + `residencias.csv`, gera `edges_antenna.parquet` e
  `antennas.parquet` na pasta de output da cidade, retorna paths gerados.

Este módulo só é necessário quando os parquets ainda não existem.

### 4.3 `src/pipeline/topology.py`

Extrair do notebook 2:

- `run(G, G_main, config, exporter)`: calcula e exporta:
  - Distribuição de grau + CCDF
  - Componentes (tamanhos)
  - Clustering médio
  - Centralidades (grau, força, betweenness amostrado, autovetor)
  - Comunidades Louvain + modularidade
  - Figuras: histograma de grau, CCDF, mapa de comunidades
  - Métricas resumidas em dict

### 4.4 `src/pipeline/spatial.py`

Extrair do notebook 3:

- `run(G_main, antennas_df, edges_antenna_df, config, exporter)`: calcula e exporta:
  - Voronoi georreferenciado
  - Homofilia socioeconômica (observada vs. acaso + matriz 5×5)
  - Decaimento de intensidade por distância
  - Rede agregada por antena (145 nós, fluxos)
  - Hubs no mapa
  - Figuras correspondentes

### 4.5 `src/pipeline/advanced.py`

Extrair do notebook 4:

- `run(G_main, config, exporter)`: calcula e exporta:
  - Ajuste de lei de potência (MLE Clauset, implementação própria)
  - Assortatividade de grau + k-nn(k)
  - Decomposição k-core
  - Coeficiente small-world (σ) vs. Erdős–Rényi
  - Curva de robustez (ataque dirigido vs. falha aleatória)
  - Figuras correspondentes

### 4.6 `src/exporter.py`

Centraliza toda saída:

```python
class Exporter:
    def __init__(self, city: str, base_dir: str, config: dict): ...
    def save_figure(self, fig, name: str, subfolder: str): ...
    def save_data(self, df, name: str): ...
    def save_metrics(self, metrics: dict, name: str = "metrics.json"): ...
    def write_report(self, sections: dict): ...  # gera report.md
```

### 4.7 `main.py` — entrypoint CLI

Usar `argparse` ou `click`:

```
usage: main.py [-h] --city CITY [--analyses {all,eda,topology,spatial,advanced} [...]
               ] [--output OUTPUT] [--no-basemap] [--config CONFIG]

Argumentos:
  --city       nome da cidade (deve ter config/<cidade>.yaml)
  --analyses   quais análises rodar (default: all)
  --output     pasta de saída (sobrescreve config)
  --no-basemap não baixar tiles do contextily (modo offline)
  --config     path para um yaml de config extra (merge sobre o default)
```

---

## 5. `src/utils.py`

Funções utilitárias compartilhadas:

- `load_config(city, extra_config_path=None) -> dict`: lê `default.yaml`, faz merge com
  `<cidade>.yaml` e com config extra se fornecida.
- `setup_logging(level="INFO")`: configura logging padrão.
- `ensure_dirs(paths: list[str])`: cria pastas de output se não existirem.

---

## 6. Relatório automático (`output/<cidade>/summary/report.md`)

O `Exporter.write_report()` deve gerar um relatório em markdown com:

1. **Cabeçalho:** cidade, data de execução, versão do pipeline.
2. **Tabela de métricas principais:** nós, arestas, densidade, componente gigante (%), clustering,
   modularidade, α (scale-free), assortatividade, k-core máximo, σ (small-world).
3. **Achados por seção:** um parágrafo por módulo com os números reais.
4. **Tradução para gestão urbana:** a tabela do CLAUDE.md § 5, preenchida com os valores reais da
   cidade analisada.

---

## 7. `requirements.txt` e `pyproject.toml`

Consolidar as dependências reais do projeto:

```
networkx>=3.6
pandas>=2.0
geopandas
shapely
scipy
numpy>=2.0
seaborn
matplotlib
contextily
pyarrow
pyyaml
click           # ou argparse (stdlib)
```

Garantir que o `pyproject.toml` declare o pacote `src` corretamente para imports funcionarem.

---

## 8. `.gitignore` — adições necessárias

```
output/
dados/residencias.csv
dados/edges.csv
*.pyc
__pycache__/
.python-version
```

---

## 9. Ordem de implementação recomendada

1. Reorganizar pastas (mover notebooks, criar `src/`, `config/`).
2. Implementar `src/utils.py` e `config/default.yaml` + `config/campinas.yaml`.
3. Implementar `src/graph_builder.py` (extraído do setup dos NBs 2–4).
4. Implementar `src/exporter.py`.
5. Implementar `src/pipeline/eda.py`.
6. Implementar `src/pipeline/topology.py`.
7. Implementar `src/pipeline/spatial.py`.
8. Implementar `src/pipeline/advanced.py`.
9. Implementar `main.py`.
10. Atualizar `requirements.txt` e `pyproject.toml`.
11. Testar com `python main.py --city campinas --analyses all`.
12. Atualizar `README.md` com instruções de uso do pipeline.

---

## 10. Critérios de aceitação

- [ ] `python main.py --city campinas --analyses all` roda do início ao fim sem erro.
- [ ] Todos os arquivos de output esperados (§ 1) são gerados.
- [ ] `python main.py --city campinas --analyses topology` roda apenas o módulo de topologia.
- [ ] `python main.py --city campinas --no-basemap` roda sem precisar de internet.
- [ ] Adicionar uma nova cidade requer apenas criar `config/<cidade>.yaml` (sem alterar código).
- [ ] Os notebooks originais em `notebooks/` continuam executando sem alteração.
- [ ] `report.md` gerado contém os números reais da cidade analisada.

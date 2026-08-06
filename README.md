# Caracterização de Redes Telefônicas Urbanas por Métodos de Redes Complexas

Pipeline modular e configurável para analisar a **rede de comunicação entre as regiões de qualquer
cidade**, a partir de dados anonimizados de chamadas cruzados com residência e quintis
socioeconômicos. A análise caracteriza a estrutura espacial e social da cidade (corredores de fluxo,
macro-regiões funcionais, gravidade, segregação, robustez) e exporta figuras, métricas e um
relatório por cidade.

> **A unidade de análise é a antena, não a pessoa.** Cada antena é uma **região** da cidade; os
> moradores sob ela entram como atributos agregados do nó, e as chamadas entre dois moradores da
> mesma antena viram a **insularidade** daquela região — não uma aresta.
>
> A rede resultante é pequena e densa (em Campinas: 145 nós, densidade 0,56), o que torna sem
> sentido métricas que dependem de cauda de grau — lei de potência, small-world contra
> Erdős–Rényi, k-core, componente gigante. Elas foram substituídas por análises de **peso**,
> **espaço** e **backbone**. Veja a tabela em [`CLAUDE.md`](CLAUDE.md#5-o-que-saiu-da-análise-e-por-quê).

## Pipeline de produção

```bash
python main.py --city campinas --analyses all
python main.py --city all                             # roda todas as cidades configuradas
python main.py --city campinas --analyses topology spatial
python main.py --city campinas --analyses advanced --output ./resultados
python main.py --city campinas --no-basemap          # modo offline (sem tiles)
```

Cidades já configuradas: **campinas, lavras, cabofrio, divinopolis, fortaleza**.

> ⚠️ **Fortaleza** tem ~750 mil usuários. A agregação por antena reduz muito o custo das análises
> (a rede de regiões é sempre pequena), mas a **EDA inicial** — expandir a base e cruzar com o
> `residencias.csv` de ~1 GB — continua lenta na primeira execução. Depois disso os parquets por
> antena ficam em cache em `dados/`.

> **Como interpretar os resultados:** veja [`GUIA_INTERPRETACAO.md`](GUIA_INTERPRETACAO.md), escrito
> para quem vai analisar os dados (explica cada figura, métrica e tabela).

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
├── data/      antenna_nodes.csv, antenna_flows.csv, backbone_flows.csv,
│              gravity_top_residuals.csv, edges_antenna.parquet, antennas.parquet
├── figures/
│   ├── topology/   strength_distribution.png, backbone.png, score.png,
│   │               balance_insularity.png
│   ├── spatial/    voronoi_quintile.png, macro_regions_map.png, insularity_map.png,
│   │               net_balance_map.png, calls_per_user_map.png, flows_all.png,
│   │               flows_backbone.png, gravity_model.png, gravity_residuals.png,
│   │               homophily.png
│   └── advanced/   robustness.png, rich_club.png, homophily_levels.png
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
│   ├── antenna.py            # rede de regiões: nós, fluxos, backbone, gravidade, s-core
│   ├── graph_builder.py      # agregação de pares de usuários (insumo da rede de regiões)
│   ├── exporter.py           # salva figuras, métricas (JSON) e relatório (Markdown)
│   ├── utils.py              # load_config, logging, criação de pastas
│   └── pipeline/
│       ├── eda.py            # gera os parquets por antena (a partir do residencias.csv)
│       ├── topology.py       # força, backbone, macro-regiões, s-core, balanço
│       ├── spatial.py        # Voronoi, corredores, gravidade, homofilia, insularidade
│       └── advanced.py       # robustez ponderada, rich-club, individual vs regional
├── config/                   # default.yaml + um yaml por cidade
├── notebooks/                # camada narrativa fina sobre src/ (mesmos números do pipeline)
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

## Notebooks (`notebooks/`)

São a **versão didática** da análise: constroem a rede passo a passo usando as primitivas de
`src/antenna.py` (os algoritmos vêm do mesmo código do pipeline, sem reimplementação), com texto
explicando cada conceito antes e interpretando os números depois. Basta trocar a variável `CIDADE`
na célula de preparação.

- `1-eda.ipynb` — EDA da base bruta, construção das tabelas por antena e caracterização da nova
  unidade de análise (tamanho das regiões, quintis, comunicação interna vs. externa).
- `2-rede-antenas.ipynb` — a rede de regiões passo a passo, **a demonstração de por que lei de
  potência, small-world e k-core perdem o sentido nessa escala**, e o que entra no lugar: força e
  desigualdade, backbone por disparidade, macro-regiões, s-core, reciprocidade.
- `3-analise-espacial.ipynb` — Voronoi e mapas temáticos, corredores, macro-regiões no mapa,
  modelo de gravidade e resíduos, homofilia por quintil.
- `4-analises-avancadas.ipynb` — robustez por eficiência ponderada, rich-club e a **falácia
  ecológica**: por que a homofilia de 1,9× não sobrevive a um modelo nulo correto.

## Análises incluídas

**Estrutura da rede de regiões** — distribuição de força, desigualdade de volume (Lorenz/Gini),
**backbone por filtro de disparidade** (Serrano et al.), **macro-regiões funcionais** (Louvain
ponderado), **s-core** (núcleo-periferia por força), reciprocidade e balanço emissor/receptor.

**Espacial e socioeconômica** — Voronoi por quintil, mapas temáticos de insularidade, balanço
líquido, chamadas por morador e macro-regiões; mapa de todos os corredores e do backbone;
**modelo de gravidade** (`F ≈ (n_i n_j)^a / d^b`) com mapa dos **resíduos**; **homofilia por quintil
ponderada por volume**, matriz de mistura e **índice de auto-preferência por estrato**.

**Avançadas** — **robustez por eficiência ponderada** (a rede densa não fragmenta; ela degrada),
**rich-club ponderado** por força e a comparação **individual vs. regional**, que separa preferência
social de proximidade territorial (falácia ecológica).

Todas as análises têm **guardas de robustez**: em cidades pequenas as que não fazem sentido são
puladas com aviso, sem quebrar a execução.

## Próximos passos sugeridos

- Comparar cidades usando os relatórios por cidade (as métricas agora são comparáveis entre redes de
  tamanhos diferentes, por serem baseadas em peso e não em contagem de nós).
- Cruzar as macro-regiões funcionais com as divisões administrativas oficiais.
- Confirmar a abrangência geográfica dos dados (ver o caveat da RMC em [`CLAUDE.md`](CLAUDE.md#8-notas-técnicas-reprodutibilidade)).
- Incorporar métricas temporais, caso exista base com timestamps das chamadas.

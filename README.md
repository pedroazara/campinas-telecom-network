# Caracterização de uma Rede Telefônica Urbana por Métodos de Redes Complexas

Este repositório caracteriza a rede telefônica urbana da cidade de **Campinas (SP)** usando
métodos de redes complexas, partindo de uma base de chamadas agregada por emissor e cruzando com
dados de residência e quintis socioeconômicos dos usuários.

## Estrutura

- `1-eda.ipynb` — análise exploratória da base `Campinas.parquet`: estrutura e qualidade dos dados,
  comportamento dos emissores, volume e concentração de chamadas (Pareto, CCDF), distância
  residencial das arestas, e construção/exportação da tabela de arestas emissor-receptor. Ao final,
  cruza com `residencias.csv` para gerar as tabelas por antena (`edges_antenna.parquet`,
  `antennas.parquet`), restritas a Campinas.
- `2-rede-complexa.ipynb` — **topologia** da rede: construção do grafo, densidade, distribuição de
  grau e CCDF, componentes conexas, clustering, **centralidades e hubs** e comunidades (Louvain).
- `3-analise-espacial.ipynb` — **estrutura espacial e socioeconômica** sobre o mapa de Campinas:
  localização das antenas, diagrama de Voronoi, visualização da rede, distribuição espacial das
  comunidades, decaimento da comunicação com a distância, homofilia socioeconômica por quintil e
  rede agregada de fluxo entre antenas. Tem um *setup* que reconstrói a rede a partir dos parquets,
  então roda de forma independente.
- `4-analises-avancadas.ipynb` — **métricas estruturais de redes complexas**: ajuste de lei de
  potência (rede livre de escala), assortatividade de grau e decomposição em k-core, análise
  small-world (vs. grafo aleatório) e robustez a ataques/falhas. Também roda de forma independente.

## Dados (`dados/`)

- `Campinas.parquet` — base principal agregada por emissor residente em Campinas. Cada linha é um
  emissor, com totais de chamadas e listas por receptor (IDs, nº de chamadas, distância, duração).
- `residencias.csv` — base de residências dos usuários (~1 GB), com cidade e quintis
  socioeconômicos. Não versionada (ver `.gitignore`); usada apenas no fim do `1-eda.ipynb`.
- `edges_antenna.parquet` / `antennas.parquet` — saídas do `1-eda.ipynb` consumidas pelo
  `2-rede-complexa.ipynb`.

## Como executar

Abra os notebooks no Jupyter/JupyterLab/VS Code com o ambiente `sistemas-complexos` e execute as
células em ordem. Os notebooks `2-rede-complexa.ipynb`, `3-analise-espacial.ipynb` e
`4-analises-avancadas.ipynb` dependem apenas dos parquets pequenos (`edges_antenna.parquet` +
`antennas.parquet`) e podem ser rodados de forma independente; o `1-eda.ipynb` precisa de
`residencias.csv` na etapa final.

Dependências: `pandas`, `numpy`, `pyarrow`, `matplotlib`, `seaborn`, `networkx`, `geopandas`,
`shapely`, `scipy`, `contextily` (este baixa o mapa de fundo, exige internet).

## Análises incluídas

**EDA (`1-eda.ipynb`)**
- Dicionário e checagens de qualidade (nulos, duplicados, consistência das listas internas).
- Distribuições de chamadas, receptores distintos e duração; Pareto e CCDF.
- Distância residencial das arestas e concentração de contatos por emissor.
- Verificação dos nós sem localização e construção das tabelas por antena.

**Topologia da rede (`2-rede-complexa.ipynb`)**
- Grafo não-direcionado ponderado, densidade, distribuição de grau e CCDF.
- Componentes conexas e componente gigante; clustering.
- **Centralidades (grau, força, intermediação, autovetor) e identificação de hubs.**
- Comunidades por Louvain e modularidade.

**Análise espacial (`3-analise-espacial.ipynb`)**
- Estrutura geográfica sobre o mapa de Campinas: scatter de antenas, Voronoi por quintil.
- Distribuição espacial das comunidades por antena e mapa de hubs.
- **Decaimento da intensidade de chamadas com a distância residencial.**
- **Homofilia socioeconômica por quintil (observado vs. modelo nulo) e matriz de mistura.**
- **Rede agregada entre antenas: fluxo de chamadas entre regiões da cidade.**

**Análises avançadas (`4-analises-avancadas.ipynb`)**
- **Ajuste de lei de potência** da distribuição de grau (rede livre de escala?).
- **Assortatividade de grau** e **decomposição em k-core** (núcleo-periferia).
- **Small-world**: clustering e caminho médio vs. grafo aleatório equivalente (coeficiente σ).
- **Robustez**: fragmentação da rede sob ataque dirigido vs. falha aleatória.

## Próximos passos sugeridos

- Comparar Campinas com outras cidades, se houver parquets equivalentes.
- Incorporar métricas temporais, caso exista base com timestamps das chamadas.
- Aprofundar a relação entre homofilia socioeconômica e segregação espacial.

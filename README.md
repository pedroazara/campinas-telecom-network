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
- `2-rede-complexa.ipynb` — construção da rede e análise de redes complexas: distribuição de grau,
  CCDF, componentes conexas, clustering, **centralidades e hubs**, comunidades (Louvain) e toda a
  estrutura espacial e socioeconômica da rede sobre o mapa de Campinas.

## Dados (`dados/`)

- `Campinas.parquet` — base principal agregada por emissor residente em Campinas. Cada linha é um
  emissor, com totais de chamadas e listas por receptor (IDs, nº de chamadas, distância, duração).
- `residencias.csv` — base de residências dos usuários (~1 GB), com cidade e quintis
  socioeconômicos. Não versionada (ver `.gitignore`); usada apenas no fim do `1-eda.ipynb`.
- `edges_antenna.parquet` / `antennas.parquet` — saídas do `1-eda.ipynb` consumidas pelo
  `2-rede-complexa.ipynb`.

## Como executar

Abra os notebooks no Jupyter/JupyterLab/VS Code com o ambiente `sistemas-complexos` e execute as
células em ordem. O `2-rede-complexa.ipynb` depende apenas dos parquets pequenos
(`edges_antenna.parquet` + `antennas.parquet`); o `1-eda.ipynb` precisa de `residencias.csv` na
etapa final.

Dependências: `pandas`, `numpy`, `pyarrow`, `matplotlib`, `seaborn`, `networkx`, `geopandas`,
`shapely`, `scipy`, `contextily` (este baixa o mapa de fundo, exige internet).

## Análises incluídas

**EDA (`1-eda.ipynb`)**
- Dicionário e checagens de qualidade (nulos, duplicados, consistência das listas internas).
- Distribuições de chamadas, receptores distintos e duração; Pareto e CCDF.
- Distância residencial das arestas e concentração de contatos por emissor.
- Verificação dos nós sem localização e construção das tabelas por antena.

**Rede complexa (`2-rede-complexa.ipynb`)**
- Grafo não-direcionado ponderado, densidade, distribuição de grau e CCDF.
- Componentes conexas e componente gigante; clustering.
- **Centralidades (grau, força, intermediação, autovetor) e identificação de hubs.**
- Comunidades por Louvain e sua distribuição espacial por antena.
- Estrutura geográfica sobre o mapa de Campinas: scatter de antenas, Voronoi por quintil.
- **Decaimento da intensidade de chamadas com a distância residencial.**
- **Homofilia socioeconômica por quintil (observado vs. modelo nulo) e matriz de mistura.**
- **Rede agregada entre antenas: fluxo de chamadas entre regiões da cidade.**

## Próximos passos sugeridos

- Comparar Campinas com outras cidades, se houver parquets equivalentes.
- Incorporar métricas temporais, caso exista base com timestamps das chamadas.
- Aprofundar a relação entre homofilia socioeconômica e segregação espacial.

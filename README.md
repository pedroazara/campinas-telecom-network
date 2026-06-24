# Caracterização de uma Rede Telefônica Urbana por Métodos de Redes Complexas

Este repositorio organiza uma analise exploratoria inicial de dados telefonicos com foco em usuarios residentes em Lavras.

## Arquivos

- `Lavras.parquet`: base principal ja agregada por emissor residente em Lavras. Cada linha representa um emissor e contem totais de chamadas, quantidade de receptores distintos e listas com informacoes por receptor.
- `residencias.csv`: base grande de residencias dos usuarios, com cidade e quintis socioeconomicos. Por ser um arquivo de quase 1 GB, o notebook trabalha com amostras e leitura em partes.
- `eda.ipynb`: notebook documentado com analises exploratorias, checagens de qualidade e sugestoes de proximos passos.

## Como executar

Abra `eda.ipynb` no Jupyter Notebook, JupyterLab ou VS Code e execute as celulas em ordem.

Dependencias usadas no notebook:

- `pandas`
- `numpy`
- `pyarrow`
- `plotly`
- `networkx`

Observacao sobre o ambiente local: durante a inspecao apareceu incompatibilidade entre `numpy` 2.x e alguns pacotes compilados opcionais (`matplotlib`, `numexpr` e `bottleneck`). Por isso, o notebook usa `plotly.graph_objects` para visualizacao e bloqueia os opcionais problemáticos antes de importar `pandas`.

## Analises incluidas

- Dicionario de dados e estrutura do parquet.
- Checagens de qualidade: nulos, duplicados, consistencia entre `unique_receivers` e listas de receptores.
- Estatisticas descritivas de chamadas e contatos.
- Distribuicoes de chamadas, receptores distintos e duracao.
- Pareto de emissores por volume de chamadas.
- Expansao das listas do parquet para uma tabela de arestas emissor-receptor.
- Analise de distancia das chamadas e duracao total.
- Analise de concentracao de contatos por emissor.
- Rede direcionada emissor-receptor com metricas basicas de grau.
- Amostra da base `residencias.csv` para distribuicao de cidades e quintis.

## Proximos passos sugeridos

- Filtrar a base de residencias para conectar os quintis aos emissores e receptores do parquet.
- Criar metricas temporais, caso exista uma base com timestamps das chamadas.
- Comparar Lavras com outras cidades, se houver parquets equivalentes.
- Avaliar comunidades na rede quando houver maior cobertura de ligacoes entre usuarios observados.

# CLAUDE.md — Contexto do projeto para o agente de IA

> Este arquivo serve para um agente de IA **interpretar o projeto** e ajudar a sintetizar uma
> apresentação. Ele resume objetivo, dados, notebooks e **todos os achados com os números reais**
> já computados, além de traduzir os resultados técnicos em **questões relevantes para a cidade**.

---

## 1. Objetivo e enquadramento da apresentação

**Tema:** caracterizar a **rede telefônica urbana de Campinas (SP)** por métodos de **redes
complexas**, a partir de dados anonimizados de chamadas cruzados com residência e quintis
socioeconômicos dos usuários.

**Formato da apresentação (importante):** o professor quer que o grupo apresente **como se ele fosse
o prefeito de Campinas**. Portanto, o produto final não é "uma lista de métricas de grafo", e sim
**questões relevantes para a gestão da cidade** sustentadas pelos dados. Cada achado técnico deve ser
traduzido em uma pergunta/insight que um prefeito acharia útil (desigualdade social, organização
territorial, infraestrutura crítica, resiliência, onde investir).

**Pergunta-guia:** *o que os padrões de comunicação telefônica revelam sobre a estrutura social e
espacial de Campinas, e o que isso sugere para políticas públicas?*

---

## 2. Dados (`dados/`)

| Arquivo | O que é | Tamanho |
|---|---|---|
| `Campinas.parquet` | Base agregada por emissor residente em Campinas; cada linha tem totais de chamadas e listas por receptor (IDs, nº de chamadas, distância residencial, duração). | pequeno |
| `residencias.csv` | Residência de cada usuário: `ID`, `residence_geometry` (ponto em WKB), `residence_city`, `residence_quintile_state`, `residence_quintile_nation`. **~1 GB, não versionado.** | grande |
| `edges.csv` | Arestas emissor→receptor expandidas da base (saída do NB1). | 56.139 arestas |
| `edges_antenna.parquet` | Arestas filtradas (só usuários com residência conhecida) + id da antena de cada extremo. | 48.947 arestas, 8 colunas |
| `antennas.parquet` | Antenas residenciais distintas em Campinas, com cidade e quintis. | 145 antenas |

**Números da base trabalhada:** 22.688 emissores e 22.894 receptores distintos; 594.892 chamadas no
total; quintis socioeconômicos `q1`–`q5` (q1 = 20% mais pobres, q5 = 20% mais ricos). Cada "antena"
corresponde a uma geometria residencial distinta e funciona como unidade espacial (~bairro/região).
**~7,8% dos nós** não têm residência cadastrada e são removidos da rede espacial (são nós periféricos
de grau baixo, mediana 3 — não hubs).

---

## 3. Estrutura dos notebooks

O ambiente Python é o conda **`sistemas-complexos`**. Os notebooks 2, 3 e 4 dependem apenas dos
parquets pequenos e **rodam de forma independente** (cada um reconstrói o grafo no seu *setup*); o
NB1 precisa do `residencias.csv`.

1. **`1-eda.ipynb`** — análise exploratória da base e construção das tabelas por antena
   (`edges_antenna.parquet`, `antennas.parquet`).
2. **`2-rede-complexa.ipynb`** — topologia: grafo, grau, CCDF, componentes, clustering,
   centralidades/hubs, comunidades (Louvain).
3. **`3-analise-espacial.ipynb`** — estrutura espacial/socioeconômica sobre o mapa de Campinas.
4. **`4-analises-avancadas.ipynb`** — métricas estruturais de redes complexas (scale-free,
   assortatividade/k-core, small-world, robustez).

**Construção do grafo (comum aos NBs 2–4):** rede **não-direcionada e ponderada**; chamadas A→B e
B→A são agregadas no mesmo par; o peso é `log1p(q_calls) * log1p(duração_total)`. As análises
estruturais usam a **componente gigante** `G_main`.

---

## 4. Achados completos (números já validados)

### 4.1 Topologia (NB2)
- Grafo: **25.176 nós, 31.509 arestas**, densidade ≈ 1×10⁻⁴ (rede muito esparsa).
- Distribuição de grau concentrada: **mediana 2**, 75% dos nós com grau ≤ 3, **grau máximo 56** →
  cauda longa (muitos pouco conectados, poucos hubs).
- Componentes: **2.549 componentes**; **componente gigante com 18.043 nós (71,7%)**.
- Clustering médio **0,16** (muito acima do aleatório).
- Comunidades (Louvain): **426 comunidades**, a maior com 369 usuários, **modularidade 0,983**
  (estrutura fortemente modular — a rede se divide em muitos grupos coesos).
- Centralidades calculadas: grau, força (grau ponderado por chamadas), intermediação (betweenness,
  amostrado k=500), autovetor — usadas para identificar e mapear hubs.

### 4.2 Espacial e socioeconômica (NB3)
- 145 antenas georreferenciadas em Campinas (coordenadas ≈ -47,05 / -22,94); diagrama de **Voronoi**
  define a "região de influência" de cada antena, colorida por quintil.
- **Comunidades têm forte concentração espacial**: cada comunidade ocupa poucas antenas e tem raio
  médio pequeno; a fração de arestas internas às comunidades fica **muito acima** de um modelo nulo.
- **Decaimento com a distância:** a intensidade média de chamadas **cai conforme aumenta a distância
  residencial** entre os usuários (efeito de gravidade espacial).
- **Homofilia socioeconômica (achado central):** **49% das chamadas ligam pessoas do mesmo quintil**,
  contra **26% esperado ao acaso** (modelo nulo) → **≈ 1,9×**. As pessoas se comunicam
  preferencialmente dentro do próprio estrato socioeconômico. (Há também a matriz de mistura 5×5 por
  quintil.)
- **Rede agregada entre antenas:** agregando usuários por antena, obtém-se uma rede de **145 regiões
  com 5.817 fluxos** (densidade 0,557), revelando os principais **corredores de chamadas entre
  regiões** da cidade.
- Mapa de hubs e visualização da rede com os usuários distribuídos **dentro das células de Voronoi**
  das suas antenas (com as fronteiras das regiões desenhadas).

### 4.3 Métricas estruturais avançadas (NB4) — componente gigante, sem self-loops (18.043 nós / 26.159 arestas)
- **Lei de potência (scale-free):** distribuição de grau de **cauda pesada**; ajuste por MLE
  (Clauset) dá **α ≈ 3,57** (x_min = 20, KS = 0,12). Faixa típica de redes sociais; potência *pura*
  não é perfeita (corpo provavelmente lognormal) — comportamento comum em redes de comunicação.
- **Assortatividade de grau:** **r = +0,40** → rede **assortativa** (hubs se conectam com hubs);
  k_nn(k) crescente. Típico de redes sociais (≠ redes tecnológicas, em geral disassortativas).
- **k-core:** núcleo máximo **k = 12**, contendo apenas **29 nós** → periferia grande + núcleo coeso
  pequeno.
- **Small-world:** clustering real **0,164** vs **0,0002** no aleatório (≈ 800×); caminho médio
  **12,1** vs **9,1** no aleatório; **σ ≈ 596 (≫ 1)** → rede **small-world** (alto agrupamento local
  com caminhos curtos). O caminho um pouco maior que o aleatório reflete a forte modularidade.
- **Robustez:** sob **ataque dirigido** aos maiores hubs a componente gigante **fragmenta com
  ~15–20% de remoção**; sob **falha aleatória** ela resiste (mantém >10% mesmo com 50% removidos) →
  **assinatura clássica de rede com hubs: robusta a falhas, frágil a ataques.**

---

## 5. Tradução para questões de cidade (para a apresentação ao "prefeito")

| Achado técnico | Questão relevante para a cidade |
|---|---|
| Homofilia socioeconômica ≈ 1,9× | **Segregação social na comunicação:** as pessoas falam quase o dobro dentro do próprio quintil. Campinas tem "bolhas socioeconômicas" que se comunicam pouco entre si — relevante para políticas de integração e mobilidade social. |
| Decaimento com a distância + comunidades geográficas | **A cidade funciona por regiões locais:** a comunicação é predominantemente de curta distância e os grupos sociais são territorialmente concentrados — informa transporte, descentralização de serviços e planejamento por bairro. |
| Hubs + robustez (frágil a ataque) | **Infraestrutura/atores críticos:** poucos nós sustentam a conectividade da rede social; sua perda fragmenta a cidade. Útil para resiliência (telecom, comunicação em emergências) e identificação de pontos focais. |
| Rede entre antenas (corredores) | **Onde estão os fluxos:** os principais corredores de chamadas entre regiões indicam eixos de interação — apoio a decisões de infraestrutura e priorização de investimento. |
| Distribuição de grau muito desigual | **Desigualdade de conectividade:** poucos muito conectados, muitos pouco conectados — dimensão social/digital da desigualdade. |

---

## 6. Sugestão de narrativa (5 atos)

1. **Quem é Campinas nos dados** — tamanho da rede, cobertura, o que cada nó/antena representa.
2. **A cidade se comunica localmente** — decaimento com a distância + comunidades concentradas no
   mapa (Voronoi).
3. **A cidade é socialmente segmentada** — homofilia socioeconômica de ≈ 1,9× (gráfico observado vs
   acaso + matriz de mistura). *Este é o achado de maior apelo "de prefeito".*
4. **A cidade tem pontos críticos** — hubs (centralidades no mapa) + curva de robustez (ataque vs
   falha).
5. **Que tipo de cidade-rede é esta** — síntese: rede social de cauda pesada, modular, small-world e
   assortativa; o que isso implica para políticas públicas.

---

## 7. Estado de completude e possíveis aprofundamentos

**Já completo:** EDA + construção de dados; topologia (grau, componentes, clustering, centralidades,
comunidades); espacial (Voronoi, comunidades no mapa, decaimento, homofilia socioeconômica, rede de
antenas, hubs); avançado (scale-free, assortatividade, k-core, small-world, robustez). Todos os
notebooks executam de ponta a ponta no env `sistemas-complexos`, com figuras embutidas.

**Aprofundamentos que fortaleceriam a apresentação (opcionais):**
- Homofilia socioeconômica **por região do mapa** (onde a segregação é maior?).
- Quintil dos **hubs** (os mais conectados são de quais estratos?).
- Modelo de **gravidade** formal (intensidade ~ f(distância, tamanho)) ou ajuste do decaimento.
- Small-world com **modelo de configuração** (preservando a sequência de graus) além do Erdős–Rényi.
- **Rich-club** (os hubs formam um clube?).
- Tabela final de **"top corredores entre regiões"** com nomes de bairros, se houver geocodificação.

---

## 8. Notas técnicas (reprodutibilidade)

- **Ambiente:** conda `sistemas-complexos` (numpy 2.4, pandas 3.0, networkx 3.6, geopandas, shapely,
  scipy, contextily, seaborn). `contextily` baixa o basemap → precisa de internet.
- **Determinismo:** Louvain e amostragens usam `seed=42`/`random_state` fixos; os números acima
  reproduzem ao reexecutar.
- **Caveats:** betweenness e caminho médio são **amostrados** (rede grande); o pacote `powerlaw` não
  está instalado, então o ajuste scale-free é uma implementação MLE (Clauset) própria; há 18
  self-loops (autochamadas) removidos nas métricas estruturais; as células de mapa do NB3 baixam
  tiles (mais lentas).
- **Para refazer tudo:** rodar `1-eda` (gera os parquets, exige `residencias.csv`) e depois 2→3→4 em
  qualquer ordem.

---

## 9. Glossário rápido

- **Componente gigante:** maior subconjunto de nós todos alcançáveis entre si (aqui, 72% da rede).
- **Comunidade / modularidade:** grupos densamente conectados internamente; modularidade alta (0,98)
  = divisão muito nítida.
- **Centralidade / hub:** medida de importância de um nó; hub = nó muito central/conectado.
- **Homofilia:** tendência de se conectar com semelhantes (aqui, mesmo quintil socioeconômico).
- **Assortatividade:** correlação de grau entre vizinhos (positiva = hubs com hubs).
- **k-core:** maior subgrafo onde todo nó tem grau ≥ k (mede núcleo/coesão).
- **Small-world (σ):** alto clustering + caminhos curtos vs. rede aleatória; σ ≫ 1 confirma.
- **Quintil:** faixa socioeconômica (q1 = 20% mais pobres … q5 = 20% mais ricos).

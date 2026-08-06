# Guia de interpretação dos dados gerados

Este guia ensina a **ler e interpretar** o que o pipeline gera para cada cidade. Escrito para quem
vai analisar os resultados — não é preciso rodar o código nem entender Python.

O pipeline caracteriza a **rede de chamadas** de uma cidade: cada **nó** é um usuário, cada **aresta**
liga duas pessoas que se telefonaram. A partir daí, medimos como essa rede social é organizada
(quem fala com quem, quão agrupada, quem são os mais conectados, como se relaciona com renda e
território).

---

## 1. Como rodar e o que sai

```bash
python main.py --city campinas        # uma cidade
python main.py --city all             # todas as cidades configuradas
python main.py --city lavras --no-basemap   # sem mapa de fundo (offline)
```

Cada execução cria a pasta `output/<cidade>/`:

```
output/<cidade>/
├── data/       tabelas (parquet/csv) com os dados por antena e rankings
├── figures/    as figuras (topology / spatial / advanced)
└── summary/    metrics.json (todos os números) + report.md (resumo pronto)
```

**Comece sempre por `summary/report.md`** — ele traz os números principais e um parágrafo por seção.
Depois use este guia para aprofundar cada figura/métrica.

---

## 2. Conceitos-chave (glossário rápido)

| Termo | O que é |
|---|---|
| **Nó** | um usuário de telefone. |
| **Aresta** | existe entre duas pessoas que se telefonaram. |
| **Grau** | quantas pessoas distintas um usuário chama (nº de contatos). |
| **Componente gigante** | o maior "bloco" de pessoas todas conectadas entre si (direta ou indiretamente). |
| **Hub** | usuário muito conectado (grau alto) — ponto central da rede. |
| **Comunidade** | grupo de pessoas que falam muito entre si e pouco com o resto. |
| **Antena** | região residencial (≈ bairro); cada usuário pertence a uma. |
| **Quintil** | faixa de renda: **q1 = 20% mais pobres … q5 = 20% mais ricos**. |
| **Modelo nulo / acaso** | um embaralhamento aleatório usado para comparar: se o valor real é muito diferente do acaso, ele é significativo. |

---

## 3. Métricas principais (`summary/metrics.json`)

O arquivo `metrics.json` tem 4 blocos: `graph`, `topology`, `spatial`, `advanced`. Abaixo, o que
cada número significa e **como interpretá-lo**.

### 3.1 `graph` — tamanho e forma geral
| Campo | Significado | Como ler |
|---|---|---|
| `nodes` / `edges` | nº de usuários / de conexões | tamanho da rede. |
| `density` | quão "cheia" a rede é (0 a 1) | quase sempre baixíssima (~0,0001): as pessoas têm poucos contatos ante o total possível. |
| `n_components` | nº de blocos desconexos | muitos blocos = rede fragmentada. |
| `giant_fraction_pct` | % de usuários no bloco principal | **alto (>60%)** = cidade bem conectada; **baixo** = amostra esparsa/fragmentada (comum em cidades com poucos usuários na base). |

### 3.2 `topology` — organização da rede
| Campo | Significado | Como ler |
|---|---|---|
| `degree_mean` / `degree_median` / `degree_max` | contatos: média / mediana / máximo | mediana baixa + máximo alto = **poucos hubs, muita gente pouco conectada**. |
| `avg_clustering` | tendência a formar "triângulos" (meus contatos se conhecem) | 0 = nenhum; ~0,1–0,2 é típico e **muito acima do acaso** → há grupos sociais reais. |
| `n_communities` / `largest_community` | nº de grupos / tamanho do maior | muitos grupos pequenos = tecido social pulverizado. |
| `modularity` | quão nítida é a divisão em grupos (0 a 1) | **> 0,7** = divisão muito clara em comunidades. |

### 3.3 `spatial` — território e renda
| Campo | Significado | Como ler |
|---|---|---|
| `n_antennas` | nº de regiões/antenas da cidade | poucas antenas (< ~15) tornam as análises por antena grosseiras. |
| `homophily_observed` | fração de chamadas dentro do **mesmo quintil** | quanto as pessoas falam com quem tem renda parecida. |
| `homophily_null` | o mesmo, esperado **ao acaso** | referência de comparação. |
| `homophily_ratio` | observado ÷ acaso | **> 1 = segregação socioeconômica** (ex.: 1,9× = falam quase o dobro dentro do próprio estrato). |
| `degree_by_quintile` | grau médio por faixa de renda | os mais ricos têm mais contatos? (desigualdade de conectividade). |
| `hub_quintile_distribution` | de quais quintis são os hubs | se q4/q5 dominam, **os mais conectados são os mais ricos**. |
| `antenna_network_density` | densidade da rede entre antenas | **= 1,0 significa que todas as regiões se conectam** (só informativo em cidades com muitas antenas). |
| `median_dominant_antenna_share` / `median_effective_antennas` / `median_community_radius_km` | quão territorial é cada comunidade | share alto + poucas antenas efetivas + raio pequeno = **comunidades = bairros**; o oposto = grupos espalhados pela cidade. |

### 3.4 `advanced` — que "tipo" de rede é
| Campo | Significado | Como ler |
|---|---|---|
| `powerlaw_alpha` | expoente da cauda da distribuição de grau | 2–4 = típico de rede social livre de escala (poucos hubs, muitos nós fracos). *Ausente se a cidade for pequena demais para estimar.* |
| `assortativity` | hubs se ligam a hubs? (−1 a +1) | **> 0 = assortativa** (hubs com hubs, típico de redes sociais). |
| `kcore_max` / `kcore_size` | núcleo mais coeso e seu tamanho | núcleo pequeno + grau alto = poucos usuários muito entrelaçados. |
| `smallworld_sigma` (σ) | combina clustering alto + caminhos curtos | **σ ≫ 1 = rede "mundo pequeno"** (grupos coesos conectados por poucos passos). |
| `attack_collapse_fraction` | % de hubs a remover para a rede fragmentar | baixo (~0,1–0,2) = **frágil a ataques dirigidos** aos hubs. |
| `rich_club_ratio_high_k` (ρ) | os hubs formam um "clube" entre si? | **ρ > 1 = sim** (hubs muito conectados entre si além do esperado). |

> **Valores ausentes são normais em cidades pequenas.** O pipeline pula (com aviso) a análise que
> não faz sentido com poucos dados (ex.: lei de potência sem cauda, homofilia com um só quintil).

---

## 4. As figuras (`figures/`)

### `topology/`
- **degree_distribution.png** — histograma dos contatos (escala log). Barra alta na esquerda + cauda
  à direita = muita gente com poucos contatos, poucos hubs.
- **ccdf.png** — a mesma distribuição "acumulada" em log-log. Uma **reta descendente** indica cauda
  pesada (poucos hubs muito conectados).
- **communities.png** — histograma do tamanho das comunidades. Muitas barras pequenas = grupos
  sociais pequenos.

### `spatial/`
- **voronoi_map.png** — o mapa dividido em regiões (uma por antena), coloridas por quintil de renda.
  Mostra a **geografia da renda** na cidade. *(pulado se houver < 4 antenas.)*
- **community_network.png** — a rede desenhada sobre o mapa: cada ponto é um usuário (colorido pela
  sua comunidade), dentro da região da sua antena, com as ligações. **Cores misturadas** = grupos
  espalhados pela cidade; **cores em blocos** = comunidades territoriais (bairristas).
- **community_spatial.png** — quão concentradas as comunidades são (share da antena dominante e raio
  médio vs. tamanho).
- **distance_decay.png** — intensidade das chamadas vs. distância. Curva **decrescente** = as pessoas
  falam mais com quem mora perto.
- **homophily_matrix.png** — matriz 5×5: para onde vão as chamadas de cada quintil. **Diagonal forte**
  = cada faixa de renda fala principalmente consigo mesma (segregação).
- **degree_by_quintile.png** — grau médio por quintil. Barras crescentes de q1→q5 = os mais ricos têm
  mais contatos.
- **hubs_quintile.png** — de quais quintis são os hubs (vermelho) vs. a população (cinza). Se o
  vermelho pende para q4/q5, **os mais conectados são os mais ricos**.
- **antenna_network.png** — os fluxos de chamadas entre regiões (corredores mais fortes).
- **hubs_map.png** — onde moram os maiores hubs.

### `advanced/`
- **powerlaw_fit.png** — a distribuição de grau com a reta de lei de potência ajustada.
- **assortativity.png** — grau médio dos vizinhos vs. grau. **Subindo** = hubs com hubs.
- **kcore.png** — decomposição em núcleos: distribuição e tamanho do núcleo por nível k.
- **smallworld.png** — clustering e caminho médio da rede real vs. uma rede aleatória equivalente.
- **robustness.png** — a curva de fragmentação: **vermelha (ataque aos hubs)** cai rápido; **azul
  (falha aleatória)** resiste. A distância entre as curvas = vulnerabilidade aos hubs.
- **rich_club.png** — o coeficiente rich-club φ(k) real vs. aleatório e a razão ρ(k). ρ acima de 1
  no lado direito = os hubs formam um clube.

---

## 5. Tabelas (`data/`)

| Arquivo | Conteúdo |
|---|---|
| `edges_antenna.parquet` | as arestas usadas (emissor, receptor, nº de chamadas, distância, antenas). |
| `antennas.parquet` | as antenas da cidade, com quintil e localização. |
| `top_hubs.csv` | os 20 usuários mais centrais (grau, força, intermediação, autovetor). |
| `antenna_flows.csv` | os fluxos de chamadas entre pares de antenas (corredores), do maior ao menor. |
| `community_spatial.csv` | por comunidade: nº de antenas, concentração, raio médio, antenas efetivas. |

---

## 6. Cuidados ao interpretar

- **Cidades pequenas (poucos usuários/antenas)** produzem números instáveis: componente gigante
  pequena, análises por antena grosseiras, algumas métricas ausentes. Compare com cautela.
- **Cidades muito grandes (centenas de milhares de usuários, ex.: Fortaleza):** as métricas que
  exigem percorrer a rede inteira — **intermediação (betweenness)**, **small-world (σ)**,
  **robustez** e **rich-club** — são **puladas** por serem computacionalmente inviáveis nesse
  tamanho. Os campos ficam ausentes no `metrics.json` e as figuras não são geradas — **isso é
  esperado, não é erro**. Todo o resto (grau, clustering, comunidades, homofilia, k-core,
  socioeconômicas, mapas) continua completo. Os hubs, nesse caso, são identificados por **grau** e
  **força** (não por betweenness).
- **Amostragem:** intermediação (betweenness) e caminho médio são estimados por amostragem — valores
  aproximados, mas estáveis (semente fixa).
- **Modelo nulo:** sempre que houver "observado vs. acaso", o que importa é a **razão** (quão acima
  do acaso), não o valor isolado.
- **Mapas:** precisam de internet para o fundo (contextily). Sem internet, use `--no-basemap` — as
  análises continuam, só o mapa de fundo some.

---

## 7. Perguntas que os dados respondem (para a apresentação)

Pensando na apresentação "para o prefeito", cada resultado vira uma pergunta de cidade:

1. **A cidade é socialmente segregada?** → `homophily_ratio` e `homophily_matrix.png`.
2. **Os mais conectados são os mais ricos?** → `hubs_quintile.png` e `hub_quintile_distribution`.
3. **A comunicação é local (bairrista) ou espalhada?** → `distance_decay.png`, `community_network.png`
   e `median_community_radius_km`.
4. **Quem são os pontos críticos e quão frágil é a rede?** → `hubs_map.png` e `robustness.png`.
5. **Onde estão os principais eixos de fluxo?** → `antenna_network.png` e `antenna_flows.csv`.
6. **Que tipo de cidade-rede é?** → síntese: cauda pesada (`powerlaw`), modular (`modularity`),
   mundo pequeno (`sigma`), com clube de hubs (`rich_club`).

Comparando várias cidades, dá para ver **quais são mais segregadas, mais bairristas ou mais
centralizadas em hubs** — o objetivo do estudo comparativo.

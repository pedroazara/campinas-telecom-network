# Guia de interpretação dos dados gerados

Este guia ensina a **ler e interpretar** o que o pipeline gera para cada cidade. Escrito para quem
vai analisar os resultados — não é preciso rodar o código nem entender Python.

O pipeline caracteriza a **rede de comunicação entre as regiões** de uma cidade: cada **nó** é uma
**antena** (uma região, ≈ bairro) e cada **aresta** é o **fluxo de chamadas** entre duas regiões. As
pessoas não são nós: elas entram como atributos agregados da região onde moram.

> **Por que a antena e não a pessoa?** A antena é o único dado geográfico realmente disponível.
> Tratando a região como unidade, todas as análises passam a falar diretamente sobre o território —
> que é o que interessa a uma discussão de política pública.

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
├── data/       tabelas (parquet/csv) com os nós, os fluxos e os rankings
├── figures/    as figuras (topology / spatial / advanced)
└── summary/    metrics.json (todos os números) + report.md (resumo pronto)
```

**Comece sempre por `summary/report.md`** — ele traz os números principais, um parágrafo por seção e
a tabela "para o prefeito". Depois use este guia para aprofundar cada figura/métrica.

---

## 2. Conceitos-chave (glossário rápido)

| Termo | O que é |
|---|---|
| **Nó / região** | uma antena: a área da cidade onde mora um conjunto de usuários. |
| **Fluxo (aresta)** | o total de chamadas trocadas entre os moradores de duas regiões. |
| **Força** | volume total de chamadas que toca uma região (o "peso" dela na cidade). |
| **Insularidade** | fração do volume de uma região que **não sai dela** (fica entre vizinhos). |
| **Balanço líquido** | a região emite mais do que recebe? (+1 só emite, −1 só recebe). |
| **Backbone** | o esqueleto: só os fluxos estatisticamente significativos. |
| **Macro-região** | conjunto de antenas que conversam mais entre si do que com o resto. |
| **Modelo de gravidade** | previsão do fluxo a partir do tamanho das regiões e da distância. |
| **Quintil** | faixa de renda da região: **q1 = 20% mais pobres … q5 = 20% mais ricos**. |
| **Modelo nulo / acaso** | um embaralhamento aleatório usado para comparar: se o real é muito diferente do acaso, ele é significativo. |

> ⚠️ **A rede de regiões é pequena e densa.** Em Campinas: 145 nós e densidade 0,56 — mais da metade
> dos pares de regiões tem contato, e o caminho médio é 1,44. Por isso **grau não distingue nada** e
> as métricas clássicas de rede esparsa (lei de potência, small-world, k-core, componente gigante)
> não aparecem mais nos resultados: elas não teriam significado. O que informa aqui é **peso**.

---

## 3. Métricas principais (`summary/metrics.json`)

### 3.1 `topology` — estrutura da rede de regiões
| Campo | Significado | Como ler |
|---|---|---|
| `n_regions` / `n_flows` | nº de regiões / de fluxos entre elas | tamanho da rede. |
| `density` | quão "cheia" a rede é (0 a 1) | tipicamente **alta** (0,5–0,8): quase toda região fala com quase toda região. |
| `n_users` / `users_per_region_median` | moradores agregados / mediana por região | quanto mais moradores por região, mais grosseira a unidade espacial. |
| `internal_call_share` | fração de **todas** as chamadas que não saem da região | **alto (>30%) = cidade que vive no bairro.** |
| `median_insularity` | insularidade típica de uma região | complementa a métrica acima, região a região. |
| `volume_gini` | desigualdade do volume entre regiões (0 a 1) | **> 0,4 = poucas regiões concentram o tráfego.** |
| `backbone_edges` / `backbone_edge_fraction` | tamanho do esqueleto | ex.: 11% dos fluxos. |
| `backbone_weight_fraction` | volume que o esqueleto preserva | **alto com poucos fluxos = a cidade tem corredores bem definidos.** |
| `backbone_regions_covered` | regiões alcançadas pelo backbone | comparar com `naive_cut_regions_covered`: mostra que o filtro não abandona as áreas pequenas. |
| `n_macro_regions` / `modularity` | nº de macro-regiões funcionais e nitidez da divisão | **0,3–0,5 é uma divisão clara** numa rede densa (não compare com a modularidade de redes esparsas, que é sempre altíssima). |
| `score_core_size` | regiões no núcleo s-core final | o "centro de gravidade" da comunicação. |
| `reciprocity` | fração dos fluxos que voltam | perto de 1 = comunicação equilibrada nos dois sentidos. |
| `assortativity_volume` | regiões de volume parecido se ligam mais? | perto de 0 = o tamanho não define com quem se fala. |

### 3.2 `spatial` — território, gravidade e renda
| Campo | Significado | Como ler |
|---|---|---|
| `gravity_distance_exponent` (b) | quão rápido o fluxo cai com a distância | **≈ 1 = decaimento clássico** (1/distância); maior = cidade mais "fechada" em torno de si. |
| `gravity_size_exponent` (a) | quanto o tamanho das regiões puxa o fluxo | < 1 = regiões grandes falam menos do que o proporcional. |
| `gravity_r2` | quanto tamanho + distância explicam | 0,2–0,4 é típico: **a maior parte do fluxo tem outras causas** — é o que os resíduos revelam. |
| `homophily_observed` / `homophily_null` / `homophily_ratio` | volume entre regiões do mesmo quintil: real, ao acaso e a razão | **cuidado:** entre regiões essa razão costuma ser baixa (~1,1). Ver a seção 6. |
| `self_preference_index` | por quintil: volume interno observado ÷ esperado | **é a métrica que importa.** > 1 = o estrato se fecha; < 1 = se dispersa pelos outros. |
| `homophily_assortativity_weighted` | assortatividade de Newman ponderada por volume | mesma leitura, num só número. |

### 3.3 `advanced` — robustez e efeito da agregação
| Campo | Significado | Como ler |
|---|---|---|
| `efficiency_attack_collapse` | fração de regiões a remover (as de maior volume) para a eficiência cair à metade | baixo = a cidade depende muito de poucas áreas. |
| `efficiency_failure_collapse` | o mesmo, removendo ao acaso | **normalmente `null`** — a perda aleatória nunca derruba a rede. Isso é o resultado, não um erro. |
| `efficiency_gap_at_20pct` | diferença entre as duas curvas com 20% removidas | quanto maior, mais a rede depende das regiões centrais. |
| `rich_club_ratio_high_strength` (ρ) | as regiões de maior tráfego falam entre si? | **ρ > 1 = sim**, formam um clube. |
| `individual_homophily_ratio_naive` | homofilia no nível da pessoa, nulo que embaralha quintil **entre pessoas** | o número "clássico" (~2×) — **e enganoso**. |
| `individual_homophily_ratio_spatial_null` | o mesmo, nulo que embaralha quintil **entre regiões** | o número honesto. Ver seção 6. |

> **Valores ausentes são normais em cidades pequenas.** O pipeline pula (com aviso) a análise que não
> faz sentido com poucos dados.

---

## 4. As figuras (`figures/`)

### `topology/`
- **strength_distribution.png** — à esquerda, o volume por região; à direita, a **curva de Lorenz**.
  Quanto mais a curva se afasta da diagonal, mais o tráfego está concentrado em poucas regiões.
- **backbone.png** — dois painéis. O da esquerda mostra que poucos fluxos guardam a maior parte do
  volume. O da direita é o argumento do método: o filtro de disparidade mantém **todas** as regiões
  representadas, enquanto um corte pelo peso bruto abandonaria as pequenas.
- **score.png** — quantas regiões sobrevivem conforme se exige mais força. O patamar final é o núcleo.
- **balance_insularity.png** — distribuição do balanço emissor/receptor e da insularidade.

### `spatial/`
- **voronoi_quintile.png** — o mapa dividido em regiões, coloridas por quintil de renda: a
  **geografia da renda** na cidade.
- **macro_regions_map.png** — as macro-regiões funcionais no mapa. **Se saírem em blocos contíguos**,
  a divisão detectada só pelos fluxos coincide com a divisão territorial — resultado forte, porque o
  algoritmo não sabe nada sobre geografia.
- **insularity_map.png** — quais regiões falam mais consigo mesmas. Escuro = bairro que "se basta".
- **net_balance_map.png** — vermelho emite mais, azul recebe mais. Sugere onde há concentração de
  serviços/trabalho (recebe) vs. áreas majoritariamente residenciais (emite).
- **calls_per_user_map.png** — intensidade de uso por morador, controlando o tamanho da região.
- **flows_all.png** / **flows_backbone.png** — todos os corredores vs. só os estruturantes. O segundo
  é o que se leva para a apresentação.
- **gravity_model.png** — à esquerda, o decaimento com a distância em log-log; à direita, previsto vs.
  observado. A dispersão em torno da diagonal é exatamente o que o próximo mapa explora.
- **gravity_residuals.png** — os pares que falam **muito mais** do que tamanho e distância
  explicariam: laços entre bairros que sugerem dependências de trabalho ou origem em comum.
- **homophily.png** — à esquerda, para onde vai o volume de cada quintil; à direita, o **índice de
  fechamento por estrato** (acima de 1 = se fecha; abaixo = se dispersa).

### `advanced/`
- **robustness.png** — eficiência de comunicação conforme regiões são removidas. **Vermelha** (perde
  as maiores) cai; **azul** (aleatória) resiste. Note que nenhuma das curvas despenca: a rede densa
  **degrada, não fragmenta**.
- **rich_club.png** — φ real vs. pesos embaralhados, e a razão ρ. ρ acima de 1 à direita = as regiões
  mais ativas concentram volume entre si.
- **homophily_levels.png** — **a figura mais importante do projeto.** Três barras: o observado, o
  acaso embaralhando quintil entre pessoas e o acaso embaralhando quintil entre regiões. Ver a
  seção 6.

---

## 5. Tabelas (`data/`)

| Arquivo | Conteúdo |
|---|---|
| `antenna_nodes.csv` | uma linha por região: moradores, volumes, insularidade, balanço, quintil, macro-região, nível s-core. |
| `antenna_flows.csv` | todos os fluxos entre regiões: chamadas, nº de pares de pessoas, distância, intensidade. |
| `backbone_flows.csv` | só os corredores estruturantes, do maior ao menor — **a lista de "onde estão os eixos da cidade"**. |
| `gravity_top_residuals.csv` | os pares que mais superam a previsão da gravidade. |
| `edges_antenna.parquet` / `antennas.parquet` | os dados de entrada usados, para rastreabilidade. |

---

## 6. O ponto mais delicado: homofilia e falácia ecológica

Uma versão anterior deste projeto concluía: *"49% das chamadas ligam pessoas do mesmo quintil, contra
26% esperado ao acaso — 1,9× mais"*, e lia isso como **segregação socioeconômica**.

O problema está no modelo nulo. Ele embaralha o quintil **entre pessoas** — e, ao fazer isso, destrói
também o fato de que vizinhos compartilham o quintil simplesmente por morarem no mesmo lugar (o
quintil é um atributo da residência). Como a comunicação é fortemente local, o "excesso" medido era,
em grande parte, apenas **proximidade geográfica**.

Refazendo a conta com um nulo que embaralha o quintil **entre regiões**, preservando quem mora com
quem, a razão praticamente desaparece (em Campinas: de **2,11×** para **1,04×**).

**Como apresentar isso:** não é um resultado negativo, é um resultado mais forte. A desigualdade da
comunicação **tem endereço**: integrar estratos sociais não é uma política sobre indivíduos, é sobre
**conectar territórios**. E o que sobrevive à correção é ainda mais concreto — o índice de
auto-preferência mostra as regiões **q5 se fechando** e as **q1 se dispersando** pelos demais
estratos.

---

## 7. Cuidados ao interpretar

- **Não compare estas métricas com as de uma rede de usuários.** Densidade, modularidade e caminho
  médio têm escalas completamente diferentes numa rede de 145 nós densa. Uma modularidade de 0,40
  aqui é uma divisão nítida; numa rede esparsa de 25 mil nós, 0,40 seria fraco.
- **Cidades pequenas (poucas antenas)** produzem números instáveis: com menos de ~15 regiões, o
  Voronoi e as macro-regiões ficam grosseiros.
- **Tamanho das regiões varia muito** (em Campinas, de 11 a 587 moradores). Prefira sempre as
  métricas normalizadas — `intensity` nos fluxos e `calls_per_user` nos nós — quando a pergunta não
  for sobre volume absoluto.
- **Falácia ecológica:** correlações entre regiões não valem para pessoas. Se um resultado for
  apresentado como afirmação sobre indivíduos, ele precisa da checagem da seção 6.
- **Modelo nulo:** sempre que houver "observado vs. acaso", o que importa é **qual acaso** — como a
  seção 6 deixa claro, a escolha do nulo pode mudar a conclusão inteira.
- **Mapas:** precisam de internet para o fundo (contextily). Sem internet, use `--no-basemap`.

---

## 8. Perguntas que os dados respondem (para a apresentação)

1. **A cidade vive no bairro ou se atravessa?** → `internal_call_share` e `insularity_map.png`.
2. **Onde estão os eixos estruturantes?** → `flows_backbone.png` e `backbone_flows.csv`.
3. **Quais são as regiões funcionais reais da cidade?** → `macro_regions_map.png` — e compare com as
   divisões administrativas oficiais.
4. **A segregação da comunicação é social ou territorial?** → `homophily_levels.png` e a seção 6.
5. **Quais estratos se fecham e quais se dispersam?** → `self_preference_index` e `homophily.png`.
6. **Que bairros falam mais do que deveriam entre si?** → `gravity_residuals.png` e
   `gravity_top_residuals.csv`.
7. **Quais regiões são críticas para a resiliência?** → `robustness.png` e `rich_club.png`.
8. **Onde as pessoas emitem e onde recebem?** → `net_balance_map.png` (pista sobre concentração de
   trabalho e serviços).

Comparando várias cidades, dá para ver **quais são mais bairristas, mais concentradas em corredores
ou mais desiguais no volume por região**.

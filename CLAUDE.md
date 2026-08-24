# CLAUDE.md — Contexto do projeto para o agente de IA

> Este arquivo serve para um agente de IA **interpretar o projeto** e ajudar a sintetizar uma
> apresentação. Ele resume objetivo, dados, código e **todos os achados com os números reais**
> já computados, além de traduzir os resultados técnicos em **questões relevantes para a cidade**.

---

## 1. Objetivo e enquadramento da apresentação

**Tema:** caracterizar a **rede de comunicação entre as regiões de Campinas (SP)** por métodos de
redes complexas, a partir de dados anonimizados de chamadas cruzados com residência e quintis
socioeconômicos.

**Mudança de unidade de análise (importante):** o projeto começou com **a pessoa** como nó. Por
orientação do professor, a unidade passou a ser a **antena** — o único dado geográfico realmente
disponível. Cada antena é uma **região da cidade**; as pessoas que moram sob ela entram como
**atributos agregados** do nó. Consequência direta: as chamadas entre dois moradores da mesma antena
deixam de ser arestas e viram a **insularidade** daquela região.

**Formato da apresentação:** o professor quer que o grupo apresente **como se ele fosse o prefeito de
Campinas**. O produto final não é uma lista de métricas de grafo, e sim **questões de gestão urbana**
sustentadas pelos dados.

**Pergunta-guia:** *o que os fluxos de chamadas entre regiões revelam sobre a organização espacial e
social de Campinas, e o que isso sugere para políticas públicas?*

---

## 2. Dados (`dados/`)

| Arquivo | O que é |
|---|---|
| `<Cidade>.parquet` | Base agregada por emissor residente na cidade; listas por receptor (IDs, nº de chamadas, distância residencial, duração). |
| `residencias.csv` | Residência de cada usuário: `ID`, `residence_geometry` (ponto em WKB), `residence_city`, `residence_quintile_state/nation`. **~1 GB, não versionado.** |
| `<cidade>_edges_antenna.parquet` | Arestas usuário→usuário (só quem tem residência conhecida) + id da antena de cada extremo. Gerado pelo módulo de EDA. |
| `<cidade>_antennas.parquet` | Antenas residenciais distintas, com cidade e quintis. |
| `limites-cidade/<cidade>_fua.geojson` | Limite territorial da cidade (GHS-FUA), ~8 KB, versionado. Extraído do geopackage global do GHSL por `scripts/extrair_limite.py`. |

**Números da base de Campinas:** 25.176 usuários com residência conhecida, distribuídos em
**145 antenas** (mediana de **148 moradores por antena**, mín. 11, máx. 587). Quintis `q1`–`q5`
(q1 = 20% mais pobres). **~7,8% dos usuários** não têm residência cadastrada e ficam de fora.

**O quintil é um atributo da antena, não da pessoa:** ele vem colado à geometria residencial, então
todos os moradores de uma antena compartilham o mesmo quintil. Isso é decisivo para interpretar a
homofilia (seção 4.3). Distribuição das 145 antenas: q5=66, q4=41, q3=21, q1=10, q2=7.

---

## 3. Estrutura do código

O pipeline é a fonte da verdade; os notebooks são uma camada narrativa fina sobre ele.

```
src/antenna.py            constrói a rede de regiões (nós, fluxos, backbone, gravidade, s-core)
src/boundary.py           carrega o limite territorial (GHS-FUA) que recorta o Voronoi
src/graph_builder.py      utilitários de agregação de pares de usuários
src/exporter.py           figuras, métricas (JSON) e relatório (MD)
src/pipeline/eda.py       gera os parquets por antena a partir do residencias.csv
src/pipeline/topology.py  força, backbone, macro-regiões, s-core, balanço
src/pipeline/spatial.py   Voronoi, corredores, gravidade, homofilia, insularidade
src/pipeline/advanced.py  robustez ponderada, rich-club, individual vs regional
main.py                   CLI: python main.py --city campinas --analyses all
painel.py                 menu no terminal: roda pipeline, scripts e notebooks sem digitar
notebooks/                1-eda, 2-rede-antenas, 3-analise-espacial, 4-analises-avancadas
```

**Construção da rede (em `src/antenna.py`):**
- **Nó** = antena. Atributos: `n_users`, `calls_out/in/total`, `calls_internal`, `insularity`,
  `net_balance`, `calls_per_user`, quintil, `lon`/`lat`.
- **Aresta** = fluxo não-direcionado entre duas antenas distintas, com `q_calls`,
  `calls_duration_total`, `n_pairs` (pares de pessoas por trás do fluxo), `dist_km` (haversine) e
  `intensity` (`q_calls` normalizado pelo produto das populações).
- Há também a versão **dirigida** (`net.D`), usada para reciprocidade e balanço emissor/receptor.
- O **peso é `q_calls` bruto** por padrão. O antigo `log1p(q_calls) * log1p(duração)` fazia sentido
  entre pessoas, não entre regiões. Configurável em `antenna.weight` (`q_calls`, `J`, `n_pairs`).

**Matriz de conexão entre antenas (`build_contact_matrix`)** — formulação do paper
*Detecting Communities from Cell Phone Antennas*:

    k_i  = Σ_l k_i(l)           contatos do indivíduo i, repartidos por antena
    K_lm = Σ_{i∈V_l} k_i(m)     contatos entre residentes de l e residentes de m
    J_lm = K_lm / (u_l · u_m)   intensidade normalizada da ligação

**A unidade é o contato, não a chamada:** dois moradores que se telefonam 200 vezes contam como
1 contato. Por isso J é uma **densidade** — a fração dos `u_l·u_m` pares possíveis que existe de
fato — e vive em [0, 1]. Isso corrige o viés de tamanho do peso bruto, em que uma região grande
sempre tem volume alto. A diagonal tem três convenções (`paper`, `density`, `zero`), porque a
definição literal conta cada par interno duas vezes e o denominador correto ali é u_l(u_l−1)/2.

---

## 4. Achados completos (números validados, `output/campinas/summary/`)

### 4.1 A rede de regiões
- **145 regiões, 5.817 fluxos, densidade 0,557** — mais da metade dos pares de regiões da cidade tem
  contato. Grau mediano 85, caminho médio **1,44**.
- **35,9% de todas as chamadas não saem da região de origem** (insularidade mediana por região: 0,18;
  máximo 0,55).
- Volume desigual entre regiões: **Gini 0,37**.
- **Backbone (filtro de disparidade, α=0,05): 656 fluxos (11% do total) carregam 62% de todas as
  chamadas** e cobrem as 145 regiões (um corte pelo peso bruto do mesmo tamanho alcançaria 141).
- **5 macro-regiões funcionais** (Louvain ponderado, modularidade 0,40, a maior com 54 antenas).
  Exportáveis como polígonos por `scripts/exportar_macroregioes.py`. **As 5 saem com uma única parte
  cada — contiguidade espacial perfeita**, e o Louvain só vê volume de chamadas. A união delas
  reproduz a cidade sem buracos nem sobreposição.
- **Insularidade da macro-região: 61% a 82%** — muito acima dos 36% por antena. Agrupadas em
  macro-regiões, as áreas se bastam: a macro-região do noroeste (Americana/Santa Bárbara, 21 antenas)
  fecha **82%** do seu volume dentro de si.
- **Atenção ao peso:** a contiguidade perfeita vale para `q_calls`. Com `J` (normalizado por
  população), **3 das 5 macro-regiões se partem** em 2–3 pedaços. A regionalização contígua é uma
  propriedade do volume, não da densidade de laços — vale dizer isso se perguntarem.
- **Reciprocidade 0,86** — quem recebe, devolve.
- Núcleo s-core final: **42 regiões**.
- **Matriz de conexão (K e J):** 23.016 contatos entre regiões e 8.493 internos; J mediano
  7,7×10⁻⁵ e máximo 5,0×10⁻³ — nenhum par de regiões chega perto de esgotar os contatos possíveis.
  **J e o volume bruto ordenam as ligações de forma bem diferente** (Spearman 0,43): as ligações
  mais fortes por J envolvem regiões pequenas (mediana de 48 moradores na menor ponta, contra 190
  quando se ordena por volume). Trocando o peso para J, o backbone cai de 656 para 427 fluxos com
  só 33% de sobreposição, e as 5 macro-regiões mudam de composição (ARI 0,53).

### 4.2 Espaço e gravidade
- Cada antena é uma **célula de Voronoi recortada pelo limite real da cidade** (GHS-FUA, seção 8);
  os mapas temáticos mostram quintil, insularidade, balanço emissor/receptor, chamadas por morador,
  densidade e macro-regiões. O contorno da cidade entra em todos os mapas, inclusive nos de fluxo.
- Com o recorte, a célula passa a ter **área**: mediana **7,7 km²** (mín. 0,15, máx. 67,4) dentro de
  uma cidade de **1.564 km²** — 0,5% de diferença contra os 1.571 km² declarados pelo próprio GHS-FUA,
  o que valida a geometria. A densidade mediana é de 23 usuários da amostra por km².
- **As 5 macro-regiões saem espacialmente contíguas**, embora o Louvain não conheça geografia — a
  divisão funcional da cidade coincide com a territorial. *É a figura de maior impacto visual.*
- **Modelo de gravidade:** `F_ij ≈ C · (n_i n_j)^0,56 / d_ij^1,09`, R² = 0,27. O expoente de
  distância ≈ 1 formaliza o decaimento (antes só mostrado como curva).
- Os **resíduos** do modelo apontam pares de bairros que falam muito mais do que tamanho e distância
  explicariam — afinidades que a geografia não capta.

### 4.3 Homofilia socioeconômica — **o achado mudou de sentido**
- **Entre regiões:** 35% do volume liga áreas do mesmo quintil, contra 31% ao acaso → **1,12×**.
- **Índice de auto-preferência por quintil** (observado/esperado) — aqui está o que sobrevive:
  **q5 = 1,23**, q4 = 0,92, **q3 = 1,35**, q1 = 0,69, q2 = 0,41. *As regiões ricas se fecham; as
  pobres se dispersam pelos demais estratos.*
- **O antigo 1,9× era um artefato do modelo nulo.** No nível individual, 58% do volume liga o mesmo
  quintil; contra um nulo que embaralha o quintil **entre pessoas**, isso dá **2,11×**. Mas esse nulo
  destrói também o fato de que vizinhos compartilham quintil por morarem no mesmo lugar. Com um nulo
  que embaralha o quintil **entre regiões** (preservando quem mora com quem), o esperado sobe para
  56% e a razão cai para **1,04×**.
- **Conclusão:** a "segregação socioeconômica na comunicação" é, em quase toda a sua extensão,
  **segregação territorial**. As pessoas falam com quem está perto, e quem está perto tem a mesma
  renda. Isso não enfraquece a apresentação — deixa o argumento mais acionável.

### 4.4 Robustez e rich-club
- **A rede não fragmenta** — ela perde capacidade aos poucos. A robustez é medida pela queda da
  **eficiência de comunicação ponderada**, não pelo tamanho da componente gigante.
- Removendo as regiões de maior volume, a eficiência cai à metade com **28%** das regiões fora; a
  perda aleatória nunca chega lá (diferença de 40 p.p. em favor do acaso com 20% removidas).
- **Rich-club ρ ≈ 2,0**: as regiões de maior tráfego concentram volume entre si muito acima do
  esperado se os pesos fossem embaralhados.

---

## 5. O que saiu da análise e por quê

A mudança de unidade invalidou boa parte do ferramental anterior. Isto é material de apresentação —
mostra domínio do método, não fracasso.

| Análise removida | Por que perdeu sentido | O que entrou no lugar |
|---|---|---|
| Lei de potência (α ≈ 3,57) | 145 nós não sustentam ajuste de cauda; grau concentrado em ~85 | desigualdade de volume (Gini/Lorenz) |
| CCDF de grau | degenera numa reta vertical | distribuição de força |
| Componente gigante / 2.549 componentes | densidade 0,56: tudo é uma componente só | — |
| Small-world (σ ≈ 596) | caminho médio já é 1,44; a aleatória equivalente também é agrupada | — |
| k-core (k=12) | trivializa em rede densa | **s-core** (poda por força) |
| Comunidades de usuários (426, Q=0,98) | comunidade agora é conjunto de antenas | **macro-regiões funcionais** (5, Q=0,40) |
| Hubs individuais no mapa | não há mais nós-pessoa | força e insularidade por região |
| Assortatividade de grau (r=+0,40) | grau não distingue nada | assortatividade **ponderada por quintil** |
| Robustez por fragmentação | a rede nunca fragmenta | **eficiência ponderada** sob remoção |

E o que a nova unidade **passou a permitir**: modelo de gravidade, resíduos de fluxo, backbone por
disparidade, insularidade, balanço emissor/receptor, reciprocidade e regionalização funcional.

---

## 6. Tradução para questões de cidade (apresentação ao "prefeito")

| Achado técnico | Questão relevante para a cidade |
|---|---|
| Segregação territorial, não social (2,11× → 1,04×) | **A desigualdade da comunicação tem endereço.** O que parecia preferência por gente da mesma renda é efeito de morar perto. Integrar estratos não é política sobre indivíduos — é **conectar territórios**. |
| q5 se fecha (1,23×), q1 se dispersa (0,69×) | **As pontas da cidade se comportam de forma oposta.** As regiões ricas concentram a comunicação em si mesmas; as pobres se espalham por todos os estratos — quem depende do resto da cidade para trabalhar, se comunica com o resto da cidade. |
| 36% das chamadas não saem da região | **A vida acontece no bairro.** Argumento direto para descentralizar serviços, saúde e equipamentos públicos. |
| Backbone: 11% dos fluxos carregam 62% do volume | **Onde investir.** A cidade tem um esqueleto de comunicação bem definido — prioridade natural para infraestrutura e redundância de telecom. |
| 5 macro-regiões funcionais e contíguas | **A cidade real vs. a cidade administrativa.** Os fluxos revelam agrupamentos de bairros que funcionam como unidade; comparar com as divisões oficiais mostra onde o desenho administrativo não acompanha a vida cotidiana. |
| Os 145 pontos de "Campinas" caem todos na área urbana funcional | **A cidade não termina na divisa.** Quem a base chama de Campinas mora também em Sumaré, Hortolândia, Valinhos, Paulínia — e se comunica como uma cidade só. O município é a unidade de governo; a **cidade funcional** é a unidade do problema. Política de mobilidade, saúde e telecom desenhada só dentro da divisa administrativa erra o alvo por construção. |
| Gravidade: fluxo cai com d^1,09 | **A distância ainda governa a interação** — e os resíduos apontam laços entre bairros distantes que indicam dependências de trabalho ou origem em comum. |
| Robustez: degradação gradual, sem colapso | **Resiliência.** A rede não se parte ao perder uma área, mas perde capacidade de forma desigual: as regiões de maior volume merecem redundância prioritária. |

---

## 7. Sugestão de narrativa (5 atos)

1. **O que estamos olhando** — 145 regiões, 25 mil moradores agregados, o que é um nó e uma aresta.
2. **A cidade tem um esqueleto** — backbone: 11% dos fluxos, 62% do volume, no mapa.
3. **A cidade se divide sozinha** — as 5 macro-regiões funcionais saem contíguas sem o algoritmo
   saber geografia. *Melhor figura da apresentação.*
4. **A segregação tem endereço** — o gráfico de três barras (observado / acaso entre pessoas / acaso
   entre regiões) e o índice de auto-preferência por quintil. *Este é o ato principal.*
5. **O que a cidade aguenta** — robustez, rich-club e a lista de regiões críticas.

---

## 8. Notas técnicas (reprodutibilidade)

- **Ambiente:** `.venv` com Python 3.13 (uv). `contextily` baixa o basemap → precisa de internet;
  use `--no-basemap` para rodar offline.
- **Execução:** `python main.py --city campinas --analyses all` (~30 s). Saída em
  `output/campinas/` (figuras, `metrics.json`, `report.md`).
- **Atalho para tudo:** `python painel.py` abre um menu que chama o pipeline, os scripts e os
  notebooks, e mostra o comando equivalente antes de cada execução.
- **Estabilidade das macro-regiões:** `python scripts/estabilidade_macroregioes.py campinas`
  responde "a semente foi escolhida a dedo?" (ARI ≥ 0,94 em 30 sementes, sempre 5 comunidades) e
  "Q=0,40 é alto?" (o nulo desta densidade dá 0,26; o observado está a ~30 desvios-padrão).
- **Macro-regiões como polígonos:** `python scripts/exportar_macroregioes.py campinas`
  (`--formato gpkg`, `--peso J`, `--seed N`). Saída em `output/<cidade>/data/macroregioes.geojson`.
- **Limite da cidade:** `python scripts/extrair_limite.py campinas` regenera o GeoJSON a partir do
  geopackage global do GHS-FUA (~10 MB, não versionado — o GeoJSON de 8 KB é). Sem o limite o
  pipeline não quebra: cai no recorte retangular antigo e avisa no log.
- **Determinismo:** Louvain, permutações e amostragens usam `seed=42`.
- **A abrangência geográfica: resolvida.** As 145 antenas estão todas marcadas com
  `residence_city == "Campinas"`, mas se espalham por **48 × 39 km**, com 46 delas a mais de 15 km do
  centro — o mapa mostra pontos em Americana, Santa Bárbara d'Oeste, Sumaré, Hortolândia, Paulínia,
  Valinhos e Jaguariúna. Isso levantou a suspeita de que ou o campo designava a região metropolitana,
  ou as geometrias eram anonimizadas de forma grosseira.
  **É a primeira hipótese.** Cruzando as antenas com o **GHS-FUA** (GHSL/OECD, release R2019A), as
  **145 de 145 caem dentro da área urbana funcional de Campinas** — nenhuma cai em FUA vizinho, nenhuma
  fica de fora. O `residence_city` designa a **região funcional**, não o município. Não há erro de
  geocodificação a corrigir, e o enquadramento certo da apresentação é o da **cidade funcional**.
- **O limite territorial (GHS-FUA).** Uma *Functional Urban Area* é o centro urbano **mais a sua zona
  de commuting** — a área de influência da cidade no mercado de trabalho —, delineada sobre uma grade
  de população de 1 km, sem depender de fronteiras administrativas. O eFUA de Campinas tem
  **1.571 km², 2,51 M habitantes (2015) e aglutina 3 centros urbanos**. É exatamente a unidade que os
  dados descrevem, então ele recorta as células de Voronoi e vira o contorno de todos os mapas — antes
  o corte era um retângulo arbitrário (bbox + 5 km) que esticava as células de borda.
  O serrilhado do contorno é a grade de 1 km da fonte, não um artefato do nosso processamento.
  As outras 4 cidades do projeto também têm FUA (Lavras 129 km², Divinópolis 284, Fortaleza 1.557,
  Cabo Frio 338), então o recorte é genérico: `config/default.yaml → spatial.boundary`, casando
  `eFUA_name` com `city_name`.
  Fonte: Schiavina M. et al., *GHSL-OECD Functional Urban Areas 2019*, EUR 30001 EN,
  doi:10.2760/67415, JRC 118845 (PDF em `limites-cidade/`).
- **Outros caveats:** o modelo nulo de homofilia é uma permutação de rótulos (não controla o espaço
  explicitamente); o R² da gravidade (0,27) é típico, mas indica que a maior parte da variação dos
  fluxos não é explicada por tamanho e distância.

---

## 9. Glossário rápido

- **Insularidade:** fração do volume de uma região que não sai dela.
- **Backbone / filtro de disparidade:** subconjunto dos fluxos estatisticamente significativos, dado
  o peso que cada nó reparte entre seus vizinhos (Serrano et al., 2009).
- **s-core:** k-core generalizado para pesos — poda por força em vez de grau.
- **Macro-região funcional:** grupo de antenas que conversam mais entre si do que com o resto.
- **Modelo de gravidade:** fluxo ≈ (tamanho × tamanho) / distância^b.
- **Rich-club ponderado:** as regiões de maior força concentram volume entre si? (ρ > 1 = sim.)
- **Eficiência ponderada:** média das inversas dos caminhos mínimos, usando 1/peso como custo.
- **Quintil:** faixa socioeconômica da região (q1 = 20% mais pobres … q5 = 20% mais ricos).
- **Falácia ecológica:** concluir sobre indivíduos a partir de dados agregados por região.
- **FUA (Functional Urban Area):** o centro urbano mais a sua zona de commuting — a cidade definida
  pela área de influência no mercado de trabalho, não pela divisa administrativa. É o limite que
  recorta as células de Voronoi aqui.

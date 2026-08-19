# Matriz de conexão entre antenas (K e J)

Implementação de `build_contact_matrix()`, em [`src/antenna.py`](src/antenna.py), seguindo
*Detecting Communities from Cell Phone Antennas* (Network Science).

Este documento registra **o que o artigo define**, **como o código implementa**, **como isso foi
verificado** e **quais decisões tiveram de ser tomadas onde o artigo é omisso**.

---

## 1. O que o artigo define

Considere uma rede em que cada nó é uma antena de telefonia e as ligações representam comunicação
entre usuários associados a antenas diferentes.

| Símbolo | Significado |
|---|---|
| `l` | a l-ésima antena |
| `u_l` | número de usuários cuja residência está associada à antena `l` |
| `V_l` | conjunto desses usuários |
| `k_i` | número de **contatos** do indivíduo `i` |

As três equações:

```
(1)   k_i  = Σ_l k_i(l)             k_i(l) = contatos de i que residem na antena l

(2)   K_lm = Σ_{i∈V_l} k_i(m)       contatos entre residentes de l e residentes de m

(3)   J_lm = K_lm / (u_l · u_m)     intensidade normalizada da ligação
```

E a interpretação, nas palavras do artigo: com `u_l = 2` e `u_m = 3`, o número máximo possível de
contatos distintos é `u_l · u_m = 6`; portanto **J_lm mede a fração de todos os pares possíveis de
usuários entre as antenas l e m que estão de fato conectados**.

### O ponto que muda tudo: contato ≠ chamada

`k_i` é o número de **contatos** — pessoas distintas com quem `i` se comunicou. Dois moradores que
se telefonam 200 vezes contam como **1 contato**, não 200.

Por isso `J` é uma **densidade**, confinada a [0, 1]: é a fração dos `u_l · u_m` pares possíveis que
existe de fato. Não é volume de tráfego.

> ⚠️ A coluna `intensity`, que já existia na tabela de fluxos deste projeto, é
> `q_calls / (u_l·u_m)` — usa **chamadas** no lugar de contatos. **Ela não é o J do artigo.** As duas
> coexistem: `J` é densidade de contato, `intensity` é volume por par possível.

---

## 2. Como o código implementa

```python
pares = build_edges_graph(edges_antenna)             # 1 registro por par de usuários conectado
pares = pares[pares["source"] != pares["target"]]    # i não é contato de i

la = pares["source"].map(user_antenna).map(posicao)  # antena de cada ponta
lb = pares["target"].map(user_antenna).map(posicao)

K = np.zeros((n, n), dtype=np.int64)
np.add.at(K, (la, lb), 1)
np.add.at(K, (lb, la), 1)

u = users.reindex(antenas).to_numpy(dtype=float)
J = K / np.outer(u, u)
```

**Por que somar nas duas posições reproduz a equação (2).** Um par conectado `{i, j}` com `i∈V_l` e
`j∈V_m` é visto pelos dois extremos: entra uma vez em `k_i(m)` e uma vez em `k_j(l)`.

- Para `l ≠ m`: `K_lm` e `K_ml` recebem +1 cada. Resultado: `K_lm = K_ml =` número de pares
  conectados entre as duas antenas — que é o que a soma da equação (2) produz.
- Para `l = m`: as duas somas caem na mesma célula, dando **+2 por par interno** — exatamente o que
  `Σ_{i∈V_l} k_i(l)` produz, já que cada par interno é contado uma vez por `i` e uma vez por `j`.

Uma regra só cobre os dois casos, sem tratamento especial.

`build_edges_graph` é o passo que garante a semântica de *contato*: ele colapsa as chamadas A→B e
B→A num único registro por par de pessoas, de modo que cada par entra na contagem uma vez,
independentemente de quantas chamadas houve.

---

## 3. Verificação

### 3.1 Contra uma implementação literal das equações

Foi escrita uma segunda implementação, deliberadamente lenta, traduzindo as equações (1)–(3) linha a
linha — montando `V_l` como conjunto, o dicionário de contatos de cada indivíduo, e iterando
`for l: for i in V_l: for j in contatos[i]`. Comparada à versão vetorizada sobre os dados reais de
Campinas (145 antenas, 25.176 usuários):

```
K idêntica à versão literal do artigo: True
J idêntica (tolerância 1e-15):         True
maior diferença em K:                  0
```

### 3.2 Contra o exemplo numérico do próprio artigo

Caso sintético com `u_l = 2`, `u_m = 3` e 4 dos 6 pares possíveis conectados:

```
K_lm = 4     J_lm = 0.6667 = 4/6   ✓
```

### 3.3 Invariantes sobre os dados reais

| Propriedade | Resultado |
|---|---|
| `K` simétrica | ✓ |
| Soma fora da diagonal ÷ 2 | 23.016 = número real de pares inter-antena ✓ |
| Soma da diagonal ÷ 2 | 8.472 = número real de pares intra-antena ✓ |
| `J = K/(u_l·u_m)` elemento a elemento | ✓ |
| `J ≤ 1` | ✓ (máximo observado: 0,073) |

O limite superior vale sempre: fora da diagonal, o número de pares não pode exceder `u_l·u_m`; na
diagonal, `K_ll ≤ u_l(u_l−1)`, então `J_ll ≤ 1 − 1/u_l < 1`.

### 3.4 Um erro que a verificação encontrou

A comparação com a versão literal revelou uma divergência de até 2 em algumas células da diagonal.
A causa: **21 autochamadas** na base (o mesmo ID nos dois extremos, 34 chamadas no total). A versão
vetorizada as somava duas vezes; a literal, uma vez.

Nenhuma das duas estava certa. O artigo define `k_i` como "o número de contatos do indivíduo `i`", e
**uma pessoa não é contato de si mesma** — autochamada é artefato do dado. Passaram a ser excluídas,
e as duas implementações então coincidem exatamente.

---

## 4. Decisões onde o artigo é omisso

O artigo trata do caso `l ≠ m` ("links represent communication between users associated with
**different** antennas"). Três pontos exigiram decisão explícita.

### 4.1 A diagonal (`l = m`)

Aplicada literalmente, a equação (2) conta cada par interno duas vezes, e o denominador `u_l·u_m`
vira `u_l²` — mas o número máximo de pares distintos dentro de uma antena é `u_l(u_l−1)/2`, não
`u_l²`. O parâmetro `diagonal` cobre as três leituras:

| Valor | `K_ll` | Denominador | Quando usar |
|---|---|---|---|
| `"paper"` (padrão) | 2 × pares internos | `u_l²` | fidelidade literal à equação (2) |
| `"density"` | pares internos | `u_l(u_l−1)/2` | quando `J_ll` deve ser comparável a `J_lm` (vira a densidade do grafo interno da região) |
| `"zero"` | 0 | — | quando só interessam as ligações entre regiões |

### 4.2 O que é "usuário" em `u_l`

O artigo diz "the number of users whose residence is associated with the l-th antenna". Isso admite
duas leituras, e **a escolha muda o valor de J**:

- **usuários da rede de chamadas** — quem aparece na base como emissor ou receptor: 25.176 em Campinas;
- **residentes cadastrados** — todas as linhas do `residencias.csv` daquela cidade: 33.455.

O padrão é o primeiro, porque `J` é definido como a fração dos pares possíveis que estão
*conectados*, e alguém que nunca aparece na base de chamadas não poderia estar. Usar a população
cadastrada reduziria `J` por um fator de aproximadamente `(25.176/33.455)² ≈ 0,57`.

A escolha é explícita, via parâmetro:

```python
cm = antenna.build_contact_matrix(edges_antenna, nodes, users=minha_serie_de_u_l)
```

### 4.3 Residência de usuários vistos sob mais de uma antena

`build_user_antenna_map` atribui a cada usuário a **moda** das antenas observadas. O artigo pressupõe
uma residência única por usuário; esta é a regra de desempate.

---

## 5. Como usar

```python
from src import antenna

cm = antenna.build_contact_matrix(edges_antenna, nodes)   # diagonal="paper"

cm.K        # DataFrame n×n de contatos (inteiros)
cm.J        # DataFrame n×n de intensidade normalizada, em [0,1]
cm.users    # a Series u_l usada como denominador
cm.flows()  # versão longa: uma linha por par conectado, com K, J, u_a, u_b
```

`J` também está disponível por aresta, sem montar a matriz:

```python
net = antenna.build(edges_antenna, antennas, config)
net.flows[["a", "b", "n_pairs", "J", "q_calls", "intensity"]]
```

E pode ser usado como **peso das arestas** do grafo, via `config/default.yaml`:

```yaml
antenna:
  weight: "J"      # padrão: "q_calls"
```

---

## 6. O que J mostra em Campinas

| | |
|---|---|
| Contatos entre regiões | 23.016 |
| Contatos internos às regiões | 8.472 |
| Pares de regiões conectados | 5.817 de 10.440 possíveis |
| `J` mediano (pares conectados) | 7,7 × 10⁻⁵ |
| `J` máximo fora da diagonal | 5,0 × 10⁻³ |

Nenhum par de regiões chega perto de esgotar os contatos possíveis — o mais denso liga 0,5% dos
pares. É o esperado numa cidade: as pessoas conhecem um punhado de gente, não bairros inteiros.

**J e o volume bruto ordenam as ligações de forma bem diferente** (correlação de Spearman **0,43**).
As ligações mais fortes por `J` envolvem regiões pequenas — mediana de 48 moradores na menor das duas
pontas, contra 190 quando se ordena por volume de chamadas. É exatamente a correção que a
normalização promete: sem ela, o mapa de fluxos apenas redesenha onde mora mais gente.

Trocar o peso de `q_calls` para `J` **não é cosmético**: o backbone cai de 656 para 427 fluxos com
apenas 33% de sobreposição, e as macro-regiões mudam de composição (ARI 0,53). Por isso o padrão
continua `q_calls` — trocar exigiria refazer todos os números já documentados.

---

## 7. Onde está no projeto

| Arquivo | Papel |
|---|---|
| [`src/antenna.py`](src/antenna.py) | `build_contact_matrix()` e o dataclass `ContactMatrix`; a coluna `J` da tabela de fluxos |
| [`src/pipeline/topology.py`](src/pipeline/topology.py) | chama, exporta as matrizes e gera o heatmap |
| `output/<cidade>/data/contact_matrix_K.csv` | matriz K, ordenada por macro-região |
| `output/<cidade>/data/contact_matrix_J.csv` | matriz J, mesma ordenação |
| `output/<cidade>/figures/topology/contact_matrix.png` | heatmap de J em escala log, agrupado por macro-região |
| [`notebooks/5-exploracao.ipynb`](notebooks/5-exploracao.ipynb) | `cm`, `K` e `J` já prontos na primeira célula |

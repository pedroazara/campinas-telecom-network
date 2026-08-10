"""Seleciona cidades pelo número de antenas (geometrias residenciais distintas).

A resolução espacial de uma cidade é o número de antenas dela: com poucas antenas, a rede
de regiões fica pequena demais para as análises do projeto. Este script varre o
``residencias.csv`` uma vez, conta as antenas distintas de cada cidade e lista as que
atingem o limiar pedido.

Exemplos:
    python scripts/cidades_por_antenas.py 100
    python scripts/cidades_por_antenas.py 50 --csv output/cidades_50.csv
    python scripts/cidades_por_antenas.py 100 --recomputar

A contagem é cacheada em ``output/antenas_por_cidade.csv``: a primeira execução lê o CSV
inteiro (~1 GB, alguns minutos) e as seguintes são instantâneas.
"""

from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESIDENCIAS_PADRAO = ROOT / "dados" / "residencias.csv"
CACHE_PADRAO = ROOT / "output" / "antenas_por_cidade.csv"

COLUNAS = ["residence_city", "residence_geometry"]


def slug(cidade: str) -> str:
    """Converte o nome da cidade no identificador usado em config/<cidade>.yaml."""
    sem_acento = unicodedata.normalize("NFKD", str(cidade))
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return "".join(c for c in sem_acento.lower() if c.isalnum())


def contar(residencias: Path, chunk: int = 1_000_000, max_linhas: int | None = None) -> pd.DataFrame:
    """Conta antenas distintas e residentes por cidade, lendo o CSV em blocos.

    Guardamos apenas os pares (cidade, geometria) já deduplicados de cada bloco — são poucos
    em relação ao número de linhas, então a memória fica sob controle mesmo num arquivo de 1 GB.
    """
    if not residencias.exists():
        raise SystemExit(f"arquivo não encontrado: {residencias}")

    pares: set[tuple[str, str]] = set()
    residentes: dict[str, int] = {}
    lidas = 0

    leitor = pd.read_csv(residencias, usecols=COLUNAS, chunksize=chunk, nrows=max_linhas)
    for bloco in leitor:
        bloco = bloco.dropna(subset=COLUNAS)
        pares.update(
            zip(bloco["residence_city"].to_numpy(), bloco["residence_geometry"].to_numpy())
        )
        for cidade, n in bloco["residence_city"].value_counts().items():
            residentes[cidade] = residentes.get(cidade, 0) + int(n)

        lidas += len(bloco)
        print(f"  {lidas:>12,} linhas | {len(residentes):>5} cidades | "
              f"{len(pares):>7,} antenas", end="\r", file=sys.stderr, flush=True)

    print(file=sys.stderr)

    antenas: dict[str, int] = {}
    for cidade, _ in pares:
        antenas[cidade] = antenas.get(cidade, 0) + 1

    tabela = pd.DataFrame({
        "cidade": list(antenas),
        "slug": [slug(c) for c in antenas],
        "antenas": [antenas[c] for c in antenas],
        "residentes": [residentes.get(c, 0) for c in antenas],
    })
    tabela["residentes_por_antena"] = (tabela["residentes"] / tabela["antenas"]).round(1)
    return tabela.sort_values("antenas", ascending=False).reset_index(drop=True)


def carregar(residencias: Path, cache: Path, recomputar: bool, chunk: int,
             max_linhas: int | None) -> pd.DataFrame:
    if cache.exists() and not recomputar and max_linhas is None:
        print(f"usando o cache de {cache} (--recomputar para refazer)", file=sys.stderr)
        return pd.read_csv(cache)

    print(f"lendo {residencias} — a primeira passagem demora alguns minutos", file=sys.stderr)
    tabela = contar(residencias, chunk=chunk, max_linhas=max_linhas)
    if max_linhas is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        tabela.to_csv(cache, index=False)
        print(f"contagem salva em {cache}", file=sys.stderr)
    return tabela


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lista as cidades com pelo menos N antenas (geometrias distintas).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("min_antenas", type=int, help="número mínimo de antenas para a cidade entrar")
    p.add_argument("--csv", default=None, help="salva a seleção neste caminho")
    p.add_argument("--recomputar", action="store_true", help="ignora o cache e relê o CSV")
    p.add_argument("--residencias", default=str(RESIDENCIAS_PADRAO), help="caminho do residencias.csv")
    p.add_argument("--cache", default=str(CACHE_PADRAO), help="onde guardar a contagem por cidade")
    p.add_argument("--chunk", type=int, default=1_000_000, help="linhas por bloco de leitura")
    p.add_argument("--max-linhas", type=int, default=None,
                   help="lê só as N primeiras linhas (teste rápido; não usa nem grava cache)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tabela = carregar(Path(args.residencias), Path(args.cache), args.recomputar,
                      args.chunk, args.max_linhas)

    selecao = tabela[tabela["antenas"] >= args.min_antenas].reset_index(drop=True)

    print(f"\n{len(selecao)} de {len(tabela)} cidades com {args.min_antenas}+ antenas\n")
    if selecao.empty:
        maior = tabela["antenas"].max() if len(tabela) else 0
        print(f"nenhuma cidade atinge o limiar (a maior tem {maior} antenas)")
        return

    w_cidade = max(len("cidade"), *(len(c) for c in selecao["cidade"]))
    w_slug = max(len("slug"), *(len(s) for s in selecao["slug"]))
    cabecalho = (f"{'cidade':<{w_cidade}}  {'slug':<{w_slug}}  {'antenas':>8} "
                 f"{'residentes':>12} {'res./antena':>12}")
    print(cabecalho)
    print("-" * len(cabecalho))
    for linha in selecao.itertuples(index=False):
        print(f"{linha.cidade:<{w_cidade}}  {linha.slug:<{w_slug}}  {linha.antenas:>8,} "
              f"{linha.residentes:>12,} {linha.residentes_por_antena:>12.1f}")

    print("\nresidentes = linhas do residencias.csv (todo mundo com residência na cidade).")
    print("Não confundir com os usuários da rede: só entram na análise quem aparece na base")
    print("de chamadas da cidade, sempre um subconjunto menor.")

    if args.csv:
        destino = Path(args.csv)
        destino.parent.mkdir(parents=True, exist_ok=True)
        selecao.to_csv(destino, index=False)
        print(f"\nseleção salva em {destino}")


if __name__ == "__main__":
    main()

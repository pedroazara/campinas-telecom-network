"""Extrai o limite de uma cidade do geopackage global do GHS-FUA e salva como GeoJSON.

O geopackage do GHS-FUA tem ~10 MB e 9.031 áreas urbanas funcionais do mundo inteiro —
peso demais para o repositório, sendo que o pipeline usa uma feição só. Este script tira
a cidade pedida e grava um GeoJSON de poucos KB em ``limites-cidade/<cidade>_fua.geojson``,
que é o arquivo versionado e lido pelo pipeline.

Exemplos:
    python scripts/extrair_limite.py campinas
    python scripts/extrair_limite.py lavras --nome-fua Lavras
    python scripts/extrair_limite.py campinas --listar

O ``--listar`` mostra as áreas do país que casam com o nome, útil quando a grafia do
GHS-FUA não bate com a de ``residence_city``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import geopandas as gpd  # noqa: E402

from config import load_config  # noqa: E402
from src.boundary import FUA_NAME_COLUMN, extract_from_gpkg, save_boundary  # noqa: E402

GPKG_PADRAO = "limites-cidade/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"


def listar(gpkg: Path, termo: str, pais: str) -> None:
    """Imprime as áreas urbanas funcionais do país cujo nome contém o termo."""
    fua = gpd.read_file(gpkg, where=f"Cntry_ISO = '{pais}'")
    achados = fua[fua[FUA_NAME_COLUMN].str.contains(termo, case=False, na=False)]
    if achados.empty:
        print(f"Nenhuma área urbana funcional em {pais} contendo '{termo}'.")
        return
    colunas = [FUA_NAME_COLUMN, "UC_num", "FUA_area", "FUA_p_2015"]
    print(achados[colunas].to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cidade", help="identificador da cidade (config/<cidade>.yaml)")
    parser.add_argument("--nome-fua", help="nome no GHS-FUA, se diferente de city_name")
    parser.add_argument("--pais", default="BRA", help="código ISO3 do país (padrão: BRA)")
    parser.add_argument("--gpkg", default=GPKG_PADRAO, help="caminho do geopackage do GHS-FUA")
    parser.add_argument("--saida", help="caminho do GeoJSON de saída")
    parser.add_argument("--listar", action="store_true", help="só lista as áreas que casam com o nome")
    args = parser.parse_args()

    config = load_config(args.cidade)
    nome = args.nome_fua or config.get("city_name", args.cidade)
    gpkg = Path(args.gpkg) if Path(args.gpkg).is_absolute() else ROOT / args.gpkg

    if args.listar:
        listar(gpkg, nome, args.pais)
        return 0

    try:
        fua = extract_from_gpkg(gpkg, nome, args.pais)
    except (FileNotFoundError, ValueError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        print("dica: rode com --listar para ver a grafia usada pelo GHS-FUA.", file=sys.stderr)
        return 1

    saida = args.saida or f"limites-cidade/{args.cidade}_fua.geojson"
    caminho = save_boundary(fua, saida)

    linha = fua.iloc[0]
    area = linha.get("FUA_area")
    pop = linha.get("FUA_p_2015")
    print(f"{linha[FUA_NAME_COLUMN]}: {area:.0f} km², {pop:,.0f} habitantes (2015), "
          f"{linha.get('UC_num', 0):.0f} centro(s) urbano(s)")
    print(f"salvo em {caminho.relative_to(ROOT)} ({caminho.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

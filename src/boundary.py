"""Limite territorial da cidade — a moldura que recorta as células de Voronoi.

Sem um limite, o diagrama de Voronoi é infinito e precisa ser cortado por um retângulo
arbitrário: as células de borda ficam esticadas até onde o `bbox` mandar, e o mapa sugere
que a cidade é um quadrado. Recortando pelo limite real, cada célula passa a ser a fatia
de território de fato atribuída àquela antena.

**A fonte é o GHS-FUA** (GHSL/OECD, release R2019A, época 2015), não um limite municipal.
Uma *Functional Urban Area* é o centro urbano **mais a sua zona de commuting** — a área de
influência da cidade no mercado de trabalho —, delineada sobre uma grade de 1 km de
população, sem depender de fronteiras administrativas.

Essa é a unidade certa para este projeto. As 145 antenas de Campinas marcadas com
``residence_city == "Campinas"`` se espalham por 48 × 39 km e caem em Sumaré, Hortolândia,
Paulínia e Valinhos — todas as 145 caem **dentro** do eFUA de Campinas (1.571 km², 2,51 M
habitantes, 3 centros urbanos aglutinados). O campo ``residence_city`` designa a região
funcional, não o município; o recorte pelo FUA torna isso explícito no mapa.

Citação da fonte: Schiavina M., Moreno-Monroy A., Maffenini L., Veneri P.,
*GHSL-OECD Functional Urban Areas 2019*, EUR 30001 EN, doi:10.2760/67415, JRC 118845.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd

logger = logging.getLogger("pipeline")

ROOT = Path(__file__).resolve().parent.parent

#: Nome da coluna do GHS-FUA que guarda o nome do centro urbano principal.
FUA_NAME_COLUMN = "eFUA_name"

#: Atributos do GHS-FUA que vale a pena carregar junto com a geometria.
FUA_FIELDS = ["eFUA_ID", "eFUA_name", "UC_num", "Cntry_ISO", "FUA_area", "FUA_p_2015", "UC_p_2015"]


def _resolve(path: str | Path) -> Path:
    """Resolve caminhos relativos contra a raiz do projeto, não contra o cwd."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def extract_from_gpkg(gpkg_path: str | Path, fua_name: str, country: str = "BRA") -> gpd.GeoDataFrame:
    """Extrai do geopackage global do GHS-FUA o polígono de uma cidade, em EPSG:4326.

    O arquivo tem 9.031 feições no mundo e ~10 MB; a leitura é filtrada no driver
    (``where``) para não carregar o globo inteiro só por causa de uma cidade.
    """
    gpkg = _resolve(gpkg_path)
    if not gpkg.exists():
        raise FileNotFoundError(f"Geopackage do GHS-FUA não encontrado em {gpkg}")

    name = str(fua_name).replace("'", "''")
    where = f"{FUA_NAME_COLUMN} = '{name}'"
    if country:
        where += f" AND Cntry_ISO = '{country}'"

    fua = gpd.read_file(gpkg, where=where)
    if fua.empty:
        raise ValueError(
            f"Nenhuma área urbana funcional chamada '{fua_name}' em {country} no GHS-FUA. "
            f"O nome deve bater com a coluna {FUA_NAME_COLUMN} (e.g. 'Campinas', 'Divinópolis')."
        )
    if len(fua) > 1:
        logger.warning(
            "[boundary] %d feições para '%s' — unindo todas num polígono só", len(fua), fua_name
        )
        fua = fua.dissolve()

    keep = [c for c in FUA_FIELDS if c in fua.columns] + ["geometry"]
    return fua[keep].to_crs(epsg=4326)


def load_boundary(config: dict, crs: str | int = "EPSG:3857") -> gpd.GeoSeries | None:
    """Carrega o limite da cidade já reprojetado, ou ``None`` se não houver.

    Ordem de busca: o GeoJSON pequeno versionado no repositório e, se ele não existir, o
    geopackage global (que não é versionado). Retorna ``None`` — sem levantar erro — quando
    nada é encontrado, para que o pipeline continue com o recorte retangular de sempre.
    """
    cfg = (config.get("spatial") or {}).get("boundary") or {}
    if not cfg.get("enabled", True):
        return None

    city = config.get("city", "cidade")
    city_name = config.get("city_name", city)

    fua = None
    geojson = cfg.get("geojson_path")
    if geojson:
        path = _resolve(str(geojson).format(city=city, city_name=city_name))
        if path.exists():
            fua = gpd.read_file(path)
            logger.info("[boundary] limite lido de %s", path.name)

    if fua is None and cfg.get("gpkg_path"):
        try:
            fua = extract_from_gpkg(
                cfg["gpkg_path"], cfg.get("fua_name") or city_name, cfg.get("country", "BRA")
            )
            logger.info("[boundary] limite extraído do geopackage do GHS-FUA")
        except (FileNotFoundError, ValueError) as exc:
            logger.warning("[boundary] limite não carregado (%s); Voronoi cai no bbox", exc)
            return None

    if fua is None or fua.empty:
        logger.warning("[boundary] nenhum limite disponível; Voronoi cai no bbox")
        return None

    geom = fua.to_crs(crs).geometry

    # Uma folga opcional evita que antenas coladas na borda percam quase toda a sua célula.
    buffer_km = float(cfg.get("buffer_km", 0.0) or 0.0)
    if buffer_km:
        geom = geom.buffer(buffer_km * 1000)

    return geom


def save_boundary(fua: gpd.GeoDataFrame, out_path: str | Path) -> Path:
    """Grava o polígono da cidade como GeoJSON (EPSG:4326, alguns KB)."""
    out = _resolve(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fua.to_crs(epsg=4326).to_file(out, driver="GeoJSON")
    return out

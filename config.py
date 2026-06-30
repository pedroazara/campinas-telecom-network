"""Configuração da cidade analisada e ponto de leitura/merge das configs.

Mantém as variáveis usadas pelos notebooks de referência (`CITY_NAME`,
`CITY_PARQUET`) e expõe `load_config`, que lê e mescla os YAMLs em `config/`
(a lógica vive em `src/utils.py`).

- CITY_NAME    → nome da cidade exatamente como aparece na coluna residence_city
                 de residencias.csv (e.g. "Campinas", "Lavras").
- CITY_PARQUET → nome do arquivo .parquet dentro da pasta dados/.
"""

CITY_NAME = "Lavras"
CITY_PARQUET = "Lavras.parquet"


def load_config(city, extra_config_path=None):
    """Lê e mescla as configs YAML da cidade (delega para src.utils.load_config)."""
    from src.utils import load_config as _load_config

    return _load_config(city, extra_config_path)

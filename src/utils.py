"""Funções utilitárias compartilhadas pelo pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

# Raiz do projeto (pai da pasta src/) e diretório de configs.
ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"

logger = logging.getLogger("pipeline")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Mescla `override` sobre `base` recursivamente, sem mutar os originais."""
    result = dict(base)
    for key, value in (override or {}).items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(city: str, extra_config_path: str | None = None) -> dict:
    """Lê `default.yaml`, mescla com `<cidade>.yaml` e com um config extra opcional.

    Adicionar uma nova cidade requer apenas criar `config/<cidade>.yaml`.
    """
    config = _read_yaml(CONFIG_DIR / "default.yaml")

    city_path = CONFIG_DIR / f"{city}.yaml"
    if not city_path.exists():
        available = sorted(p.stem for p in CONFIG_DIR.glob("*.yaml") if p.stem != "default")
        raise FileNotFoundError(
            f"Não há configuração para a cidade '{city}'. "
            f"Crie {city_path} (cidades disponíveis: {', '.join(available) or 'nenhuma'})."
        )
    config = deep_merge(config, _read_yaml(city_path))

    if extra_config_path:
        config = deep_merge(config, _read_yaml(Path(extra_config_path)))

    config.setdefault("city", city)
    config.setdefault("city_name", city.capitalize())
    return config


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configura o logging padrão do pipeline."""
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    return logger


def ensure_dirs(paths: list[str | Path]) -> None:
    """Cria as pastas de output (e intermediárias) se ainda não existirem."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)

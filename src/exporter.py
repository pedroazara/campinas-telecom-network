"""Centraliza a saída do pipeline: figuras, dados, métricas (JSON) e relatório (Markdown)."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend não-interativo (salva figuras sem abrir janela)
import matplotlib.pyplot as plt
import numpy as np

from .utils import ensure_dirs

logger = logging.getLogger("pipeline")

PIPELINE_VERSION = "1.0.0"


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


class Exporter:
    """Cuida de toda a escrita em ``output/<cidade>/``."""

    def __init__(self, city: str, base_dir: str, config: dict):
        self.city = city
        self.city_name = config.get("city_name", city.capitalize())
        self.config = config
        self.out_cfg = config.get("output", {})
        self.fmt = self.out_cfg.get("figure_format", "png")
        self.dpi = self.out_cfg.get("figure_dpi", 150)

        self.base = Path(base_dir) / city
        self.dirs = {
            "data": self.base / "data",
            "figures": self.base / "figures",
            "summary": self.base / "summary",
        }
        ensure_dirs(list(self.dirs.values()))

        self.metrics: dict[str, dict] = {}
        self.report_sections: list[tuple[str, str]] = []

    # ----------------------------------------------------------------- figuras
    def save_figure(self, fig, name: str, subfolder: str = "") -> Path | None:
        if not self.out_cfg.get("save_figures", True):
            plt.close(fig)
            return None
        folder = self.dirs["figures"] / subfolder if subfolder else self.dirs["figures"]
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.{self.fmt}"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info("figura salva: %s", path.relative_to(self.base.parent))
        return path

    # ------------------------------------------------------------------- dados
    def save_data(self, df, name: str) -> Path | None:
        if not self.out_cfg.get("save_data", True):
            return None
        path = self.dirs["data"] / name
        if str(name).endswith(".csv"):
            df.to_csv(path, index=False)
        else:
            df.to_parquet(path, index=False)
        logger.info("dados salvos: %s", path.relative_to(self.base.parent))
        return path

    # ---------------------------------------------------------------- métricas
    def add_metrics(self, section: str, metrics: dict) -> None:
        self.metrics.setdefault(section, {}).update(metrics)

    def add_report_section(self, title: str, body: str) -> None:
        self.report_sections.append((title, body))

    def save_metrics(self, name: str = "metrics.json") -> Path | None:
        if not self.out_cfg.get("save_report", True):
            return None
        path = self.dirs["summary"] / name
        payload = {
            "city": self.city,
            "city_name": self.city_name,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "pipeline_version": PIPELINE_VERSION,
            "metrics": self.metrics,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=_json_default)
        logger.info("métricas salvas: %s", path.relative_to(self.base.parent))
        return path

    # --------------------------------------------------------------- relatório
    def _m(self, section: str, key: str, default="—"):
        """Busca segura de uma métrica; formata números."""
        value = self.metrics.get(section, {}).get(key)
        if value is None:
            return default
        if isinstance(value, float):
            return f"{value:.3g}"
        return value

    def _metrics_table(self) -> str:
        rows = [
            ("Nós (rede completa)", self._m("graph", "nodes")),
            ("Arestas", self._m("graph", "edges")),
            ("Densidade", self._m("graph", "density")),
            ("Componente gigante (%)", self._m("graph", "giant_fraction_pct")),
            ("Clustering médio", self._m("topology", "avg_clustering")),
            ("Comunidades (Louvain)", self._m("topology", "n_communities")),
            ("Modularidade", self._m("topology", "modularity")),
            ("Homofilia socioeconômica (obs/acaso)", self._m("spatial", "homophily_ratio")),
            ("Expoente α (scale-free)", self._m("advanced", "powerlaw_alpha")),
            ("Assortatividade de grau", self._m("advanced", "assortativity")),
            ("k-core máximo", self._m("advanced", "kcore_max")),
            ("σ (small-world)", self._m("advanced", "smallworld_sigma")),
        ]
        out = ["| Métrica | Valor |", "|---|---|"]
        out += [f"| {label} | {value} |" for label, value in rows]
        return "\n".join(out)

    def _urban_table(self) -> str:
        ratio = self._m("spatial", "homophily_ratio")
        sigma = self._m("advanced", "smallworld_sigma")
        rows = [
            (
                f"Homofilia socioeconômica ≈ {ratio}×",
                "**Segregação social na comunicação:** as pessoas falam preferencialmente dentro do "
                "próprio quintil — relevante para políticas de integração e mobilidade social.",
            ),
            (
                "Decaimento com a distância + comunidades geográficas",
                "**A cidade funciona por regiões locais:** comunicação majoritariamente de curta "
                "distância e grupos territorialmente concentrados — informa transporte e serviços.",
            ),
            (
                "Hubs + robustez (frágil a ataque)",
                "**Infraestrutura/atores críticos:** poucos nós sustentam a conectividade; sua perda "
                "fragmenta a rede — relevante para resiliência e comunicação em emergências.",
            ),
            (
                "Rede entre antenas (corredores)",
                "**Onde estão os fluxos:** os principais corredores entre regiões indicam eixos de "
                "interação — apoio a decisões de infraestrutura e priorização de investimento.",
            ),
            (
                f"Estrutura small-world (σ ≈ {sigma})",
                "**Coesão local com alcance global:** grupos muito unidos conectados por poucos passos "
                "— informação e influência se espalham rápido através dos hubs.",
            ),
        ]
        out = ["| Achado técnico | Questão relevante para a cidade |", "|---|---|"]
        out += [f"| {a} | {b} |" for a, b in rows]
        return "\n".join(out)

    def write_report(self, sections: dict | None = None, name: str = "report.md") -> Path | None:
        if not self.out_cfg.get("save_report", True):
            return None

        if sections is not None:
            self.report_sections = list(sections.items())

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        parts = [
            f"# Relatório — Rede Telefônica de {self.city_name}",
            "",
            f"- **Cidade:** {self.city_name} (`{self.city}`)",
            f"- **Gerado em:** {now}",
            f"- **Versão do pipeline:** {PIPELINE_VERSION}",
            "",
            "## Métricas principais",
            "",
            self._metrics_table(),
            "",
            "## Achados por seção",
            "",
        ]
        if self.report_sections:
            for title, body in self.report_sections:
                parts += [f"### {title}", "", body, ""]
        else:
            parts += ["_(nenhuma seção de análise foi executada)_", ""]

        parts += [
            "## Tradução para gestão urbana",
            "",
            "Apresentação no formato “para o prefeito”: cada achado técnico vira uma questão de cidade.",
            "",
            self._urban_table(),
            "",
        ]

        path = self.dirs["summary"] / name
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(parts))
        logger.info("relatório salvo: %s", path.relative_to(self.base.parent))
        return path

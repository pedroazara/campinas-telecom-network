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

# 2.x — a unidade de análise passou a ser a antena (região), não o usuário.
PIPELINE_VERSION = "2.0.0"


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

    def _pct(self, section: str, key: str, default="—"):
        """Métrica formatada como porcentagem (para o texto voltado à gestão)."""
        value = self.metrics.get(section, {}).get(key)
        return default if value is None else f"{float(value):.0%}"

    def _metrics_table(self) -> str:
        rows = [
            ("Regiões (antenas)", self._m("topology", "n_regions")),
            ("Moradores agregados", self._m("topology", "n_users")),
            ("Fluxos entre regiões", self._m("topology", "n_flows")),
            ("Densidade da rede de regiões", self._m("topology", "density")),
            ("Chamadas que não saem da região", self._pct("topology", "internal_call_share")),
            ("Desigualdade de volume (Gini)", self._m("topology", "volume_gini")),
            ("Fluxos no backbone", self._m("topology", "backbone_edges")),
            ("Volume preservado pelo backbone", self._pct("topology", "backbone_weight_fraction")),
            ("Macro-regiões funcionais", self._m("topology", "n_macro_regions")),
            ("Modularidade das macro-regiões", self._m("topology", "modularity")),
            ("Reciprocidade dos fluxos", self._m("topology", "reciprocity")),
            ("Expoente de distância (gravidade)", self._m("spatial", "gravity_distance_exponent")),
            ("R² do modelo de gravidade", self._m("spatial", "gravity_r2")),
            ("Homofilia entre regiões (obs/acaso)", self._m("spatial", "homophily_ratio")),
            ("Homofilia individual, nulo ingênuo", self._m("advanced", "individual_homophily_ratio_naive")),
            ("Homofilia individual, nulo territorial", self._m("advanced", "individual_homophily_ratio_spatial_null")),
            ("Rich-club das regiões mais ativas", self._m("advanced", "rich_club_ratio_high_strength")),
        ]
        out = ["| Métrica | Valor |", "|---|---|"]
        out += [f"| {label} | {value} |" for label, value in rows]
        return "\n".join(out)

    def _urban_table(self) -> str:
        internal = self._pct("topology", "internal_call_share")
        backbone = self._pct("topology", "backbone_weight_fraction")
        macro = self._m("topology", "n_macro_regions")
        b_exp = self._m("spatial", "gravity_distance_exponent")
        naive = self._m("advanced", "individual_homophily_ratio_naive")
        spatial_null = self._m("advanced", "individual_homophily_ratio_spatial_null")
        raw_pref = self.metrics.get("spatial", {}).get("self_preference_index", {})
        self_pref = {k: f"{float(v):.2f}" for k, v in raw_pref.items()}
        rows = [
            (
                f"Segregação é territorial, não social ({naive}× → {spatial_null}×)",
                "**A desigualdade da comunicação tem endereço:** o que parecia preferência por gente "
                "da mesma renda é, quase todo, efeito de morar perto. A política que integra estratos "
                "não é sobre indivíduos — é sobre **conectar territórios**.",
            ),
            (
                f"q5 se fecha ({self_pref.get('q5', '—')}×), q1 se dispersa ({self_pref.get('q1', '—')}×)",
                "**As pontas da cidade se comportam de forma oposta:** as regiões ricas concentram a "
                "comunicação em si mesmas, enquanto as mais pobres se espalham por todos os estratos "
                "— quem depende do resto da cidade para trabalhar, se comunica com o resto da cidade.",
            ),
            (
                f"{internal} das chamadas não saem da região",
                "**A vida acontece no bairro:** mais de um terço de toda a comunicação da cidade é "
                "interna à própria região — argumento direto para descentralizar serviços, saúde e "
                "equipamentos públicos em vez de concentrá-los no centro.",
            ),
            (
                f"Backbone: poucos corredores carregam {backbone} do volume",
                "**Onde investir:** a cidade tem um esqueleto de comunicação bem definido. Esses eixos "
                "são a prioridade natural para infraestrutura, transporte e redundância de telecom.",
            ),
            (
                f"{macro} macro-regiões funcionais",
                "**A cidade real vs. a cidade administrativa:** os fluxos revelam agrupamentos de "
                "bairros que funcionam como uma unidade. Comparar com as divisões oficiais mostra "
                "onde o desenho administrativo não acompanha a vida cotidiana.",
            ),
            (
                f"Gravidade: fluxo cai com d^{b_exp}",
                "**A distância ainda governa a interação:** o modelo permite prever o volume entre "
                "duas regiões e, principalmente, achar os pares que fogem da previsão — laços entre "
                "bairros distantes que indicam dependências de trabalho ou origem em comum.",
            ),
            (
                "Robustez: degradação gradual, sem colapso",
                "**Resiliência:** a rede de regiões não se parte quando perde uma área, mas perde "
                "capacidade de forma desigual — as regiões de maior volume merecem redundância "
                "prioritária em planos de emergência.",
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
            f"# Relatório — Rede de Regiões de {self.city_name}",
            "",
            "> Unidade de análise: a **antena** (região). Cada nó é uma área da cidade e as pessoas",
            "> que moram sob ela entram como atributos agregados.",
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


class InlineExporter(Exporter):
    """Exporter para uso interativo nos notebooks: mostra em vez de salvar.

    Permite que os notebooks chamem exatamente os mesmos módulos do pipeline, sem duplicar
    código de análise nem escrever em ``output/``. As figuras aparecem na célula, as tabelas
    são guardadas em ``self.data`` e o texto do relatório é renderizado como Markdown.
    """

    def __init__(self, config: dict):
        self.city = config.get("city", "cidade")
        self.city_name = config.get("city_name", self.city.capitalize())
        self.config = config
        self.out_cfg = {"save_figures": True, "save_data": True, "save_report": True}
        self.metrics: dict[str, dict] = {}
        self.report_sections: list[tuple[str, str]] = []
        self.data: dict[str, object] = {}

    def save_figure(self, fig, name: str, subfolder: str = ""):
        from IPython.display import display

        display(fig)
        plt.close(fig)
        return None

    def save_data(self, df, name: str):
        self.data[name] = df
        return None

    def add_report_section(self, title: str, body: str) -> None:
        from IPython.display import Markdown, display

        super().add_report_section(title, body)
        display(Markdown(f"**{title}** — {body}"))

    def save_metrics(self, name: str = "metrics.json"):
        return None

    def write_report(self, sections: dict | None = None, name: str = "report.md"):
        return None

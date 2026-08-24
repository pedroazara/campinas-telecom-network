"""Painel de controle do projeto no terminal.

    python painel.py

Reúne num menu tudo que hoje mora em comandos avulsos: rodar o pipeline, extrair o limite da
cidade, exportar as macro-regiões, testar a estabilidade do Louvain, executar os notebooks e
abrir os resultados. O painel não implementa nenhuma análise — ele **chama** `main.py` e os
scripts de `scripts/`, e mostra o comando equivalente antes de cada execução, para que dê para
copiar e rodar na mão quando precisar.

Sem dependências além da biblioteca padrão: menu numerado, que funciona igual no PowerShell,
no cmd e num terminal Unix.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
PESOS = ["q_calls", "J", "n_pairs"]

#: O que as análises precisam para rodar. Checado uma vez, na abertura do painel.
DEPENDENCIAS = ["shapely", "geopandas", "networkx", "pandas", "matplotlib", "yaml"]


# --------------------------------------------------------------------------- interpretador
def python_do_venv() -> Path | None:
    """O Python do .venv do projeto, se existir."""
    for caminho in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if caminho.exists():
            return caminho
    return None


def reexecutar_no_venv() -> None:
    """Reabre o painel com o Python do .venv se ele foi aberto com outro.

    `python painel.py` costuma pegar o Python do PATH, que não tem geopandas nem shapely. Sem
    isto, o painel abre normalmente e só quebra lá na frente, dentro de um subprocesso, com um
    traceback que parece erro da análise e não do interpretador errado.
    """
    if os.environ.get("PAINEL_REEXEC"):          # trava contra reexecução em laço
        return
    venv = python_do_venv()
    if venv is None or Path(sys.executable).resolve() == venv.resolve():
        return
    os.environ["PAINEL_REEXEC"] = "1"
    try:
        os.execv(str(venv), [str(venv), str(Path(__file__).resolve()), *sys.argv[1:]])
    except OSError:
        pass                                      # segue no interpretador atual; avisa depois


def dependencias_faltando(interpretador: Path) -> list[str]:
    """Quais módulos das análises faltam no interpretador dado."""
    codigo = (
        "import importlib.util as u, sys; "
        f"print(' '.join(m for m in {DEPENDENCIAS!r} if u.find_spec(m) is None))"
    )
    try:
        r = subprocess.run([str(interpretador), "-c", codigo],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.split() if r.returncode == 0 else []
    except Exception:
        return []


# --------------------------------------------------------------------------- terminal
def _preparar_saida() -> bool:
    """Tenta pôr o stdout em UTF-8 e diz se dá para usar os caracteres de moldura.

    O console do Windows costuma abrir em cp1252, que não tem os box-drawing (─ │ ┌) nem
    os símbolos de status (▶ ✔ ✖) — imprimir sem checar levanta UnicodeEncodeError e derruba
    o painel. Acentos o cp1252 tem, então só os símbolos precisam de plano B.
    """
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    try:
        "─│┌┐└┘▶✔✖".encode(sys.stdout.encoding or "ascii")
        return True
    except (UnicodeEncodeError, LookupError):
        return False


UNICODE_OK = _preparar_saida()

S = {
    "h": "─" if UNICODE_OK else "-",
    "v": "│" if UNICODE_OK else "|",
    "tl": "┌" if UNICODE_OK else "+",
    "tr": "┐" if UNICODE_OK else "+",
    "bl": "└" if UNICODE_OK else "+",
    "br": "┘" if UNICODE_OK else "+",
    "run": "▶" if UNICODE_OK else ">",
    "ok": "✔" if UNICODE_OK else "OK",
    "fail": "✖" if UNICODE_OK else "X",
    "arrow": "→" if UNICODE_OK else "->",
}


def _cores_ativas() -> bool:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return False
    if os.name == "nt":                      # habilita ANSI no console do Windows
        try:
            import ctypes

            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


COR = _cores_ativas()


def c(texto: str, estilo: str) -> str:
    if not COR:
        return texto
    codigos = {"titulo": "1;36", "ok": "32", "aviso": "33", "erro": "31",
               "apagado": "90", "destaque": "1;37", "chave": "1;33"}
    return f"\033[{codigos[estilo]}m{texto}\033[0m"


def limpar() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def cidades() -> list[str]:
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml") if p.stem != "default")


class Sair(Exception):
    """Ctrl+D ou stdin fechado: encerra o painel de qualquer profundidade de menu."""


def perguntar(texto: str) -> str:
    """Lê uma linha. Ctrl+C volta ao menu; Ctrl+D (ou stdin fechado) encerra o painel."""
    try:
        return input(texto).strip()
    except KeyboardInterrupt:
        print()
        return ""
    except EOFError:
        raise Sair from None


def pausa() -> None:
    perguntar(c("\n[enter] para voltar ao painel ", "apagado"))


# --------------------------------------------------------------------------- execução
def executar(argumentos: list[str], titulo: str) -> bool:
    """Roda um subprocesso mostrando o comando equivalente e repassando a saída ao vivo."""
    interpretador = python_do_venv() or Path(sys.executable)
    cmd = [str(interpretador)] + [str(a) for a in argumentos]
    exibido = " ".join(["python"] + [str(a) for a in argumentos])
    print(f"\n{c(S['run'] + ' ' + titulo, 'titulo')}")
    print(c(f"  $ {exibido}\n", "apagado"))

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    try:
        codigo = subprocess.run(cmd, cwd=ROOT, env=env).returncode
    except KeyboardInterrupt:
        print(c("\n interrompido.", "aviso"))
        return False

    print(c(f"\n{S['ok']} concluído", "ok") if codigo == 0
          else c(f"\n{S['fail']} terminou com código {codigo}", "erro"))
    return codigo == 0


# --------------------------------------------------------------------------- estado
class Estado:
    """O que o painel carrega entre uma ação e outra."""

    def __init__(self) -> None:
        disponiveis = cidades()
        self.cidade = "campinas" if "campinas" in disponiveis else (disponiveis or ["campinas"])[0]
        self.basemap = True

    @property
    def saida(self) -> Path:
        return ROOT / "output" / self.cidade

    @property
    def limite(self) -> Path:
        return ROOT / "limites-cidade" / f"{self.cidade}_fua.geojson"

    def parquets(self) -> tuple[bool, str]:
        """Os parquets por antena já existem? (sem eles o pipeline roda a EDA, que é lenta)"""
        try:
            from src.utils import load_config

            d = load_config(self.cidade)["data"]
            ea, an = ROOT / d["edges_antenna_path"], ROOT / d["antennas_path"]
            return (ea.exists() and an.exists()), ea.name
        except Exception:
            return False, "?"

    def resumo(self) -> list[str]:
        ok_parquets, nome = self.parquets()
        linhas = [
            f"  cidade      {c(self.cidade, 'destaque')}",
            f"  basemap     {'ligado' if self.basemap else c('desligado (offline)', 'aviso')}",
            "  parquets    "
            + (c("prontos", "ok") if ok_parquets
               else c("ausentes — a EDA vai rodar (lenta)", "aviso"))
            + c(f"  ({nome})", "apagado"),
            "  limite      "
            + (c("extraído", "ok") if self.limite.exists()
               else c("não extraído — Voronoi cai no retângulo", "aviso")),
        ]
        linhas.append("  resultados  " + self._resultados())
        return linhas

    def _resultados(self) -> str:
        metrics = self.saida / "summary" / "metrics.json"
        if not metrics.exists():
            return c("nenhum ainda — rode o pipeline", "apagado")
        try:
            m = json.loads(metrics.read_text(encoding="utf-8"))["metrics"]
        except Exception as exc:
            return c(f"metrics.json ilegível ({exc})", "aviso")
        t, s = m.get("topology", {}), m.get("spatial", {})
        partes = []
        if t.get("n_macro_regions"):
            partes.append(f"{t['n_macro_regions']} macro-regiões (Q={t.get('modularity', 0):.2f})")
        if s.get("n_regions_mapped"):
            partes.append(f"{s['n_regions_mapped']} regiões")
        if s.get("city_area_km2"):
            partes.append(f"{s['city_area_km2']:,.0f} km²")
        return c(" | ".join(partes), "apagado") if partes else c("gerados", "ok")


# --------------------------------------------------------------------------- menus
def escolher(titulo: str, opcoes: list[tuple[str, str]], rodape: str = "") -> str:
    """Menu numerado genérico. Devolve a chave escolhida ou '' para voltar."""
    print(f"\n{c(titulo, 'titulo')}")
    for chave, rotulo in opcoes:
        print(f"   {c(chave, 'chave')}  {rotulo}")
    if rodape:
        print(c(f"   {rodape}", "apagado"))
    escolha = perguntar(c("\n> ", "chave"))
    if escolha in {k for k, _ in opcoes}:
        return escolha
    if escolha:
        print(c("  opção inválida.", "aviso"))
    return ""


def menu_analises(e: Estado) -> None:
    op = escolher("Quais análises rodar?", [
        ("1", "todas (all)"),
        ("2", "eda        - gera os parquets por antena a partir do residencias.csv"),
        ("3", "topology   - força, backbone, macro-regiões, s-core, matriz de contato"),
        ("4", "spatial    - Voronoi, mapas, gravidade, homofilia"),
        ("5", "advanced   - robustez, rich-club, individual vs regional"),
        ("6", "topology + spatial"),
    ], rodape="[enter] volta")
    escolhas = {"1": ["all"], "2": ["eda"], "3": ["topology"], "4": ["spatial"],
                "5": ["advanced"], "6": ["topology", "spatial"]}.get(op)
    if not escolhas:
        return
    args = ["main.py", "--city", e.cidade, "--analyses", *escolhas]
    if not e.basemap:
        args.append("--no-basemap")
    executar(args, f"pipeline: {', '.join(escolhas)}")
    pausa()


def menu_macroregioes(e: Estado) -> None:
    formato = escolher("Formato do arquivo", [
        ("1", "GeoJSON  - leve, versionável, abre em qualquer lugar"),
        ("2", "GeoPackage (.gpkg)  - formato nativo do QGIS"),
        ("3", "Shapefile (.shp)    - compatibilidade com softwares antigos"),
    ], rodape="[enter] volta")
    fmt = {"1": "geojson", "2": "gpkg", "3": "shp"}.get(formato)
    if not fmt:
        return

    peso = escolher("Peso das arestas no Louvain", [
        ("1", "q_calls  - volume bruto de chamadas (o padrão do projeto)"),
        ("2", "J        - normalizado por população; as macro-regiões deixam de ser contíguas"),
        ("3", "n_pairs  - contatos brutos"),
    ], rodape="[enter] usa o do config")
    args = ["scripts/exportar_macroregioes.py", e.cidade, "--formato", fmt]
    escolhido = {"1": "q_calls", "2": "J", "3": "n_pairs"}.get(peso)
    if escolhido:
        args += ["--peso", escolhido]
    executar(args, "exportar macro-regiões como polígonos")
    pausa()


def menu_notebooks(e: Estado) -> None:
    nbs = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not nbs:
        print(c("\nnenhum notebook encontrado.", "aviso"))
        return pausa()
    op = escolher("Executar qual notebook?", [(str(i + 1), nb.name) for i, nb in enumerate(nbs)],
                  rodape="[enter] volta - a execução sobrescreve as saídas do notebook")
    if not op:
        return
    nb = nbs[int(op) - 1]
    if perguntar(c(f"\nisso sobrescreve as saídas de {nb.name}. confirmar? [s/N] ",
                   "aviso")).lower() not in ("s", "sim", "y"):
        return
    executar(["-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
              "--ExecutePreprocessor.timeout=1800", str(nb.relative_to(ROOT))],
             f"executar {nb.name}")
    pausa()


def ver_relatorio(e: Estado) -> None:
    relatorio = e.saida / "summary" / "report.md"
    if not relatorio.exists():
        print(c(f"\nsem relatório em {relatorio.relative_to(ROOT)} - rode o pipeline antes.",
                "aviso"))
        return pausa()
    print(f"\n{c(str(relatorio.relative_to(ROOT)), 'titulo')}\n")
    print(relatorio.read_text(encoding="utf-8"))
    pausa()


def abrir_saida(e: Estado) -> None:
    if not e.saida.exists():
        print(c(f"\n{e.saida.relative_to(ROOT)} ainda não existe - rode o pipeline antes.",
                "aviso"))
        return pausa()
    try:
        if os.name == "nt":
            os.startfile(e.saida)                                   # noqa: S606
        else:
            subprocess.run(["open" if sys.platform == "darwin" else "xdg-open", str(e.saida)])
        print(c(f"\nabrindo {e.saida.relative_to(ROOT)}", "ok"))
    except Exception as exc:
        print(c(f"\nnão consegui abrir ({exc}). O caminho é:\n  {e.saida}", "aviso"))
    pausa()


def trocar_cidade(e: Estado) -> None:
    disponiveis = cidades()
    op = escolher("Cidade", [(str(i + 1), nome) for i, nome in enumerate(disponiveis)],
                  rodape="[enter] volta")
    if op:
        e.cidade = disponiveis[int(op) - 1]


# --------------------------------------------------------------------------- painel
def painel() -> int:
    if not cidades():
        print(c("Nenhuma cidade configurada em config/. Crie um config/<cidade>.yaml.", "erro"))
        return 1

    interpretador = python_do_venv() or Path(sys.executable)
    faltando = dependencias_faltando(interpretador)
    if faltando:
        print(c("\nAs análises não vão rodar: faltam módulos no interpretador.", "erro"))
        print(f"  interpretador: {interpretador}")
        print(f"  faltando:      {', '.join(faltando)}")
        if python_do_venv() is None:
            print(c("\n  Não encontrei .venv/ no projeto. Crie o ambiente:", "aviso"))
            print("    uv sync          (ou: python -m venv .venv && "
                  ".venv\\Scripts\\pip install -r requirements.txt)")
        else:
            print(c("\n  O .venv existe mas está incompleto. Reinstale as dependências:", "aviso"))
            print("    uv sync          (ou: .venv\\Scripts\\pip install -r requirements.txt)")
        if perguntar(c("\nabrir o painel mesmo assim? [s/N] ", "aviso")).lower() not in ("s", "sim", "y"):
            return 1

    e = Estado()

    while True:
        limpar()
        largura = 66
        titulo = "  Redes telefônicas urbanas " + S["h"] + " painel de controle"
        print(c(S["tl"] + S["h"] * largura + S["tr"], "titulo"))
        print(c(S["v"], "titulo") + c(titulo.ljust(largura), "destaque") + c(S["v"], "titulo"))
        print(c(S["bl"] + S["h"] * largura + S["br"], "titulo"))
        for linha in e.resumo():
            print(linha)

        op = escolher("O que você quer fazer?", [
            ("1", "Rodar o pipeline           (figuras, métricas e relatório)"),
            ("2", "Exportar macro-regiões     (polígonos para o QGIS)"),
            ("3", "Extrair o limite da cidade (GHS-FUA -> GeoJSON)"),
            ("4", "Testar estabilidade        (semente, modelo nulo, peso)"),
            ("5", "Executar um notebook"),
            ("6", "Ver o relatório"),
            ("7", "Abrir a pasta de resultados"),
            ("c", "Trocar de cidade"),
            ("b", "Ligar/desligar o basemap"),
            ("q", "Sair"),
        ])

        if op == "q":
            print(c("\naté logo.\n", "apagado"))
            return 0
        elif op == "1":
            menu_analises(e)
        elif op == "2":
            menu_macroregioes(e)
        elif op == "3":
            executar(["scripts/extrair_limite.py", e.cidade], "extrair o limite da cidade")
            pausa()
        elif op == "4":
            executar(["scripts/estabilidade_macroregioes.py", e.cidade],
                     "estabilidade das macro-regiões")
            pausa()
        elif op == "5":
            menu_notebooks(e)
        elif op == "6":
            ver_relatorio(e)
        elif op == "7":
            abrir_saida(e)
        elif op == "c":
            trocar_cidade(e)
        elif op == "b":
            e.basemap = not e.basemap


if __name__ == "__main__":
    reexecutar_no_venv()   # antes de tudo: garante que o painel roda no Python que tem as libs
    try:
        raise SystemExit(painel())
    except (KeyboardInterrupt, Sair):
        print("\naté logo.\n")
        raise SystemExit(0)

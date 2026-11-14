"""
Genera docs/diagrama_agente.png (Avance 3) leyendo los límites REALES de src/tools.py y src/agent.py.

Uso (desde la raíz del repo):  python docs/generar_diagrama_agente.py
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src import agent, tools

fig, ax = plt.subplots(figsize=(14, 8.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 8.2); ax.axis("off")

AZUL, VERDE, NARANJA, GRIS = ("#e8eefc", "#1f4fd8"), ("#e6f6ec", "#1b8a4b"), ("#fff1e0", "#d9822b"), ("#f1f1f1", "#777")


def box(x, y, w, h, t, c=AZUL, fs=8.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=c[0], ec=c[1], lw=1.4))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, weight="bold" if bold else None)


def arr(x1, y1, x2, y2, t=None, color="#444", dx=0.0, dy=0.14, both=False, fs=7.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="<->" if both else "->", color=color, lw=1.4))
    if t:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + dy, t, ha="center", fontsize=fs, color=color)


ax.text(7, 7.9, "Arquitectura del agente — TutorAE (pregunta → LLM → tool → resultado → respuesta)",
        ha="center", fontsize=12.5, weight="bold")

# --- Sistema propio (marco)
ax.add_patch(FancyBboxPatch((0.2, 0.3), 9.3, 7.2, boxstyle="round,pad=0.05", fc="none", ec="#1b8a4b", lw=1.6, ls="--"))
ax.text(0.4, 7.25, "SISTEMA PROPIO (los documentos y el índice vectorial nunca salen de aquí)", fontsize=8.5, color="#1b8a4b", weight="bold")

box(0.5, 5.7, 2.2, 1.0, "Usuario\n(chat web: Streamlit / Flask)", GRIS, bold=True)
box(3.4, 5.4, 2.9, 1.6, f"Bucle del agente\nsrc/agent.py\nmáx. {agent.MAX_ITERACIONES} iteraciones, {agent.MAX_TOOL_CALLS_PREGUNTA} tool calls\n«primero la base»", AZUL, bold=True)
box(6.9, 5.45, 2.4, 0.9, "Registro de uso\nlogs/uso_tools.jsonl\n(tools, args, llamadas, tokens)", GRIS)

box(0.5, 3.4, 2.7, 1.2, f"buscar_en_base\nmáx. {tools.MAX_FRAGMENTOS} fragmentos × {tools.MAX_CHARS_FRAGMENTO} car.\ndevuelve documento y página", VERDE)
box(3.5, 3.4, 2.7, 1.2, "generar_referencia_apa\nlistar_documentos_base\n(catálogo exacto, sin LLM)", VERDE)
box(6.5, 3.4, 2.8, 1.2, f"buscar_en_web\nmáx. {tools.WEB_MAX_RESULTADOS} resultados × {tools.WEB_MAX_CHARS_EXTRACTO} car.\nsolo tras buscar_en_base", NARANJA)

box(0.5, 1.2, 2.7, 1.2, "ChromaDB local\n303 fragmentos TOGAF 9.1\n(embeddings MiniLM en CPU)", AZUL)
box(3.5, 1.2, 2.7, 1.2, "Catálogo de documentos\n(autor, año, título, URL)", AZUL)
box(6.5, 1.2, 2.8, 1.2, "Sitios permitidos (fijos en código):\n" + ", ".join(tools.WEB_DOMINIOS_PERMITIDOS[:3]) + ",\n" + ", ".join(tools.WEB_DOMINIOS_PERMITIDOS[3:]), GRIS, fs=7.5)

# --- Servicios externos
box(10.4, 5.4, 3.3, 1.6, "LLM — Groq (externo)\nopenai/gpt-oss (tool calling)\nrecibe: pregunta + historial +\nfragmentos que devolvieron las tools", NARANJA, bold=True)
box(10.4, 2.7, 3.3, 1.6, "Buscador web — Tavily (externo)\nrecibe SOLO el texto de la consulta\n(máx. %d car.) y la lista de dominios\ndevuelve título, URL y extracto" % tools.MAX_CHARS_CONSULTA_WEB, NARANJA, bold=True)
ax.text(12.05, 1.9, "Sale del sistema:\n• al LLM: lo que ve en el prompt\n• a la web: solo la consulta\nNO sale: corpus, índice, claves", ha="center", fontsize=8, color="#b35a00")

# --- Flechas
arr(2.7, 6.2, 3.4, 6.2, "1 pregunta", both=False)
arr(3.4, 5.8, 2.7, 5.8, "6 respuesta", dy=-0.28)
arr(6.3, 6.95, 10.4, 6.95, "2 mensajes + tools disponibles", color="#d9822b")
arr(10.4, 6.6, 6.3, 6.6, "3 respuesta o tool_call", color="#d9822b", dy=0.08)
arr(4.3, 5.4, 2.0, 4.6, "4 ejecuta la tool", dx=0.9, dy=0.1)
arr(4.85, 5.4, 4.85, 4.6)
arr(5.6, 5.4, 7.9, 4.6)
arr(6.3, 5.9, 6.9, 5.9, None)
arr(1.85, 3.4, 1.85, 2.4, both=True)
arr(4.85, 3.4, 4.85, 2.4, both=True)
arr(7.9, 3.4, 7.9, 2.4, both=True)
arr(9.3, 4.0, 10.4, 3.7, None, color="#d9822b")
ax.text(9.85, 4.25, "consulta →\n← resultados", ha="center", fontsize=7.5, color="#d9822b")
ax.text(0.5, 4.85, "5 el resultado vuelve al LLM\n(ToolMessage) → paso 2", fontsize=7.5, color="#444")

fig.tight_layout()
salida = RAIZ / "docs" / "diagrama_agente.png"
fig.savefig(salida, dpi=170)
print(f"[OK] {salida}")

"""
Genera docs/diagrama_flujo_rag.png leyendo los parámetros REALES de src/config.py,
así el diagrama nunca contradice al código.

Uso (desde la raíz del repo):  python docs/generar_diagrama.py
"""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from src import config

n_pdf = len([f for f in config.DOCS_DIR.iterdir() if f.suffix.lower() in config.EXTENSIONES_SOPORTADAS])
modelo = config.GROQ_MODEL.split("/")[-1]
emb = config.EMBEDDING_MODEL.replace("paraphrase-multilingual-", "")

fig, ax = plt.subplots(figsize=(13, 7.2))
ax.set_xlim(0, 13); ax.set_ylim(-0.3, 6.9); ax.axis("off")


def box(x, y, w, h, t, c="#e8eefc", ec="#1f4fd8", fs=8.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", fc=c, ec=ec, lw=1.3))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)


def arr(x1, y1, x2, y2, t=None, dx=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", color="#444", lw=1.3))
    if t:
        ax.text((x1 + x2) / 2 + dx, (y1 + y2) / 2 + 0.12, t, ha="center", fontsize=8, color="#555")


ax.text(6.5, 6.6, "Flujo RAG — TutorAE (Arquitectura Empresarial)", ha="center", fontsize=12.5, weight="bold")

# ---------------- Indexación ----------------
ax.text(0.2, 6.05, "INDEXACIÓN  (offline · scripts/build_index.py)", fontsize=9, weight="bold", color="#1f4fd8")
xs = [0.2, 2.75, 5.3, 7.85, 10.4]
et = [
    f"Corpus\n{n_pdf} PDF de TOGAF 9.1\n(ADM, Fase Preliminar,\nVisión, Negocio, Info.)",
    "Ingesta + limpieza\nPyMuPDF · quita marca\nde agua y encabezados\nmeta: source, página",
    f"Chunking\nRecursiveCharacterSplitter\nsize {config.CHUNK_SIZE} · overlap {config.CHUNK_OVERLAP}\n+ chunk_id y sección",
    f"Embeddings\n{emb}\n(384 dim · local · CPU)",
    "ChromaDB persistente\ndistancia coseno\nmetadatos para citar",
]
for x, t in zip(xs, et):
    box(x, 4.75, 2.3, 1.15, t)
for i in range(4):
    arr(xs[i] + 2.3, 5.32, xs[i + 1], 5.32)

# ---------------- Consulta ----------------
ax.text(0.2, 4.2, "CONSULTA  (online · app.py → src/pipeline.py)", fontsize=9, weight="bold", color="#1f4fd8")
o, ob = "#fff4e5", "#d98200"
box(0.2, 2.75, 2.1, 1.1, "Pregunta del usuario\n+ historial\n(guardado en el navegador)", o, ob, 8)
box(2.7, 2.75, 2.2, 1.1, "Reformulación (LLM)\n→ pregunta\nautocontenida", o, ob)
box(5.3, 2.75, 2.2, 1.1, f"Recuperación\nsimilitud coseno\ntop_k = {config.TOP_K}", o, ob)
box(7.9, 2.75, 2.2, 1.1, f"¿Relevancia ≥ {config.MIN_RELEVANCE:.2f}?", "#fdecec", "#c0392b")
box(10.5, 2.75, 2.3, 1.1, "Expansión de vecinos\n(chunks n−1 y n+1)\n+ prompt aumentado", o, ob, 8)
for a, b in [(2.3, 2.7), (4.9, 5.3), (7.5, 7.9)]:
    arr(a, 3.3, b, 3.3)
arr(10.1, 3.3, 10.5, 3.3); ax.text(10.3, 3.43, "sí", ha="center", fontsize=8, color="#555")
ax.annotate("", xy=(6.4, 4.75), xytext=(6.4, 3.85),
            arrowprops=dict(arrowstyle="<-", color="#1f4fd8", lw=1.2, ls="--"))
ax.text(6.5, 4.3, "lee el índice", fontsize=7.5, color="#1f4fd8")

# ---------------- Salida ----------------
box(7.9, 1.15, 2.2, 1.0, "\"No encontré información\nsobre esto…\"\n(sin llamar al LLM)", "#fdecec", "#c0392b", 7.5)
arr(9.0, 2.75, 9.0, 2.15, "no", dx=0.2)
box(10.5, 1.15, 2.3, 1.0, f"LLM Groq · {modelo}\nT = {config.LLM_TEMPERATURE:g}\nsystem + few-shot + XML", "#e6f6ea", "#2e8b57", 8)
arr(11.65, 2.75, 11.65, 2.15)
box(2.8, 1.15, 4.2, 1.0, "Interfaz de chat (Flask)\nrespuesta + citas + desplegable de fuentes\n(documento · página · sección · fragmento)", "#e6f6ea", "#2e8b57", 8)
arr(7.9, 1.65, 7.0, 1.65)
ax.plot([11.65, 11.65, 4.9], [1.15, 0.6, 0.6], color="#444", lw=1.3)
ax.annotate("", xy=(4.9, 1.15), xytext=(4.9, 0.6), arrowprops=dict(arrowstyle="->", color="#444", lw=1.3))

salida = Path(__file__).resolve().parent / "diagrama_flujo_rag.png"
plt.savefig(salida, dpi=170, bbox_inches="tight")
print("Diagrama guardado en", salida)

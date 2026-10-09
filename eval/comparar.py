"""
Compara configuraciones evaluadas (antes/después) en tabla Markdown y gráfico.

Uso:  python -m eval.comparar base chunk1000
Salida: eval/resultados/comparacion.md y comparacion.png
"""

import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / "resultados"
METRICAS = [
    ("faithfulness", "Faithfulness"), ("answer_relevancy", "Answer relevancy"),
    ("context_precision", "Context precision"), ("context_recall", "Context recall"),
]
EXTRA = [
    ("hit_rate_recuperacion", "Hit-rate recuperación"),
    ("abstencion_correcta_fuera_corpus", "Abstención correcta (fuera de corpus)"),
    ("abstencion_incorrecta_dentro_corpus", "Abstención incorrecta (dentro)"),
]


def main():
    tags = sys.argv[1:]
    if len(tags) < 2:
        sys.exit("Uso: python -m eval.comparar <tag_antes> <tag_despues> [más tags]")
    res = {t: json.loads((RES / f"{t}_resumen.json").read_text(encoding="utf-8")) for t in tags}

    def val(r, clave, grupo):
        return (r.get(grupo) or {}).get(clave)

    lineas = ["| Métrica | " + " | ".join(tags) + (" | Δ (último − primero) |" if len(tags) >= 2 else " |"),
              "|---|" + "---|" * (len(tags) + 1)]
    for clave, nombre in METRICAS:
        vals = [val(res[t], clave, "ragas") for t in tags]
        fila = [f"{v:.3f}" if v is not None else "n/d" for v in vals]
        delta = f"{vals[-1] - vals[0]:+.3f}" if None not in (vals[0], vals[-1]) else "n/d"
        lineas.append(f"| {nombre} | " + " | ".join(fila) + f" | {delta} |")
    for clave, nombre in EXTRA:
        vals = [val(res[t], clave, "basicas") for t in tags]
        fila = [f"{v:.3f}" if v is not None else "n/d" for v in vals]
        delta = f"{vals[-1] - vals[0]:+.3f}" if None not in (vals[0], vals[-1]) else "n/d"
        lineas.append(f"| {nombre} | " + " | ".join(fila) + f" | {delta} |")
    lineas.append("")
    lineas.append("**Configuraciones:**")
    for t in tags:
        c = res[t]["config"]
        lineas.append(f"- `{t}`: chunk_size={c['chunk_size']}, overlap={c['chunk_overlap']}, top_k={c['top_k']}, "
                      f"min_relevance={c['min_relevance']}, embeddings={c['embedding_model']}, llm={c['llm']}")
    tabla = "\n".join(lineas)
    (RES / "comparacion.md").write_text(tabla, encoding="utf-8")
    print(tabla)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        x = np.arange(len(METRICAS))
        ancho = 0.8 / len(tags)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for i, t in enumerate(tags):
            vals = [val(res[t], c, "ragas") or 0 for c, _ in METRICAS]
            barras = ax.bar(x + i * ancho, vals, ancho, label=t)
            ax.bar_label(barras, fmt="%.2f", fontsize=8)
        ax.set_xticks(x + ancho * (len(tags) - 1) / 2, [n for _, n in METRICAS])
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Puntaje (0–1)")
        ax.set_title("Evaluación Ragas por configuración")
        ax.legend()
        fig.tight_layout()
        fig.savefig(RES / "comparacion.png", dpi=160)
        print(f"\nGráfico: {RES / 'comparacion.png'}")
    except ImportError:
        print("(instala matplotlib para generar el gráfico)")


if __name__ == "__main__":
    main()

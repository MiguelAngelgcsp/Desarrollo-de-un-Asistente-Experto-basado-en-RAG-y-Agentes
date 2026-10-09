"""
Ayuda a elegir MIN_RELEVANCE: muestra la relevancia del mejor fragmento para cada pregunta
(dentro y fuera del corpus). El umbral ideal queda entre el máximo de las "fuera" y el mínimo de las "dentro".

Uso:  python -m eval.calibrar_umbral
"""

import json
from pathlib import Path

from src.document_loader import load_documents
from src.embeddings import get_embeddings
from src.splitter import split_documents
from src.vectorstore import build_vector_store, indice_existe, load_vector_store


def main():
    emb = get_embeddings()
    vs = load_vector_store(emb) if indice_existe() else build_vector_store(split_documents(load_documents()), emb)
    preguntas = json.loads((Path(__file__).parent / "preguntas_eval.json").read_text(encoding="utf-8"))
    dentro, fuera = [], []
    for q in preguntas:
        top = vs.similarity_search_with_relevance_scores(q["pregunta"], k=1)[0][1]
        (dentro if q["en_corpus"] else fuera).append(top)
        print(f"{'DENTRO' if q['en_corpus'] else 'FUERA ':6} {top:6.3f}  {q['pregunta'][:80]}")
    print(f"\nMín. dentro del corpus: {min(dentro):.3f} | Máx. fuera del corpus: {max(fuera):.3f}")
    if min(dentro) > max(fuera):
        print(f"Umbral sugerido (punto medio): {(min(dentro) + max(fuera)) / 2:.2f}")
    else:
        print("Las distribuciones se solapan: usa un umbral bajo (≤ mín. dentro) y deja que el prompt controle la abstención.")


if __name__ == "__main__":
    main()

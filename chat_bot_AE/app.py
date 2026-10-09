"""
Interfaz web del Tutor Académico (Flask).

- El historial lo mantiene el navegador y viaja en cada petición (sin estado en el servidor):
  funciona igual con varias instancias y no se pierde si el servidor se reinicia.
- La respuesta incluye las fuentes (documento, página, sección y fragmento).

Local:   python app.py
Nube:    gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120
"""

import os

from flask import Flask, jsonify, render_template, request

from src.config import MAX_HISTORY_MESSAGES
from src.document_loader import load_documents
from src.embeddings import get_embeddings
from src.llm import get_llm
from src.pipeline import rag_pipeline
from src.splitter import split_documents
from src.vectorstore import build_vector_store, indice_existe, load_vector_store

MAX_CHARS_PREGUNTA = 1000

app = Flask(__name__)


def iniciar_rag():
    """Carga embeddings, índice y LLM UNA sola vez al arrancar."""
    embeddings = get_embeddings()
    if indice_existe():
        vector_store = load_vector_store(embeddings)
    else:  # primer arranque sin índice precalculado
        vector_store = build_vector_store(split_documents(load_documents()), embeddings, force_rebuild=False)
    return vector_store, get_llm()


vector_store, llm = iniciar_rag()


def _limpiar_historial(raw):
    historial = []
    if isinstance(raw, list):
        for m in raw[-MAX_HISTORY_MESSAGES:]:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant"):
                historial.append({"role": m["role"], "content": str(m.get("content", ""))[:2000]})
    return historial


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    pregunta = str(data.get("message", "")).strip()
    if not pregunta:
        return jsonify({"error": "Mensaje vacío"}), 400
    if len(pregunta) > MAX_CHARS_PREGUNTA:
        return jsonify({"error": f"La pregunta supera {MAX_CHARS_PREGUNTA} caracteres"}), 400

    try:
        r = rag_pipeline(pregunta, vector_store, llm, historial=_limpiar_historial(data.get("history")))
    except Exception as e:  # p. ej. límite de la API de Groq
        app.logger.exception("Error en el pipeline RAG")
        return jsonify({"error": f"No se pudo generar la respuesta: {type(e).__name__}. Intenta de nuevo."}), 500

    return jsonify(
        {
            "answer": r["respuesta"],
            "sources": r["fuentes"],
            "no_info": r["sin_informacion"],
            "search_query": r["pregunta_reformulada"],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=os.environ.get("FLASK_DEBUG") == "1")

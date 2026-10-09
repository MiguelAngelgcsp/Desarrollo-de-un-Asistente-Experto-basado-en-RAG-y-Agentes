"""
Pasos 5-7 — Recuperación + Generación, con memoria conversacional.

consulta + historial
   -> (1) reformulación a pregunta autocontenida (solo si hay historial)
   -> (2) recuperación por similitud coseno (top_k) con umbral de relevancia mínima
   -> (3) si ningún fragmento supera el umbral -> "No encontré información..." (sin llamar al LLM)
   -> (4) prompt aumentado (system + historial + contexto + pregunta) -> LLM (Groq)
"""

import os

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import MAX_HISTORY_MESSAGES, MIN_RELEVANCE, TOP_K
from src.prompts import CONDENSE_PROMPT, NO_INFO, SYSTEM_PROMPT, construir_mensaje_usuario, formatear_historial

MAX_CHARS_FRAGMENTO_UI = 600


def reformular_pregunta(pregunta: str, historial, llm) -> str:
    """Convierte una pregunta de seguimiento en una pregunta autocontenida para la búsqueda."""
    if not historial:
        return pregunta
    mensajes = [
        SystemMessage(content=CONDENSE_PROMPT),
        HumanMessage(
            content=f"<historial>\n{formatear_historial(historial)}\n</historial>\n<pregunta>{pregunta}</pregunta>"
        ),
    ]
    try:
        reescrita = llm.invoke(mensajes).content.strip().strip('"')
    except Exception:
        return pregunta  # si falla la reformulación, se busca con la pregunta original
    return reescrita or pregunta


def recuperar(pregunta: str, vector_store, k: int = TOP_K, min_relevance: float = MIN_RELEVANCE):
    """Devuelve [(Document, relevancia)] ordenado por relevancia, filtrado por umbral."""
    pares = vector_store.similarity_search_with_relevance_scores(pregunta, k=k)
    return [(d, float(s)) for d, s in pares if s >= min_relevance]

def expandir_vecinos(recuperados, vector_store, ventana: int = 1):
    """Agrega los chunks contiguos (n-1, n+1) del mismo documento."""
    ids_vistos = {d.metadata["chunk_id"] for d, _ in recuperados}
    extra = []
    for d, score in recuperados:
        fuente, n = d.metadata["chunk_id"].rsplit("::", 1)
        for delta in range(-ventana, ventana + 1):
            if delta == 0 or int(n) + delta < 0:
                continue
            vecino_id = f"{fuente}::{int(n) + delta:04d}"
            if vecino_id in ids_vistos:
                continue
            res = vector_store.get(where={"chunk_id": vecino_id})
            if res["documents"]:
                from langchain_core.documents import Document
                extra.append((Document(page_content=res["documents"][0],
                                       metadata=res["metadatas"][0]), score * 0.9))
                ids_vistos.add(vecino_id)
    todos = recuperados + extra
    # ordenar por documento y posición para que el LLM lea la lista en orden
    return sorted(todos, key=lambda p: p[0].metadata["chunk_id"])

def _fuente_para_ui(doc, score: float) -> dict:
    texto = " ".join(doc.page_content.split())
    if len(texto) > MAX_CHARS_FRAGMENTO_UI:
        texto = texto[:MAX_CHARS_FRAGMENTO_UI].rstrip() + "…"
    return {
        "documento": os.path.basename(doc.metadata.get("source", "?")),
        "pagina": doc.metadata.get("pagina"),
        "seccion": doc.metadata.get("seccion"),
        "fragmento": texto,
        "relevancia": round(score, 3),
    }


def rag_pipeline(
    pregunta: str,
    vector_store,
    llm,
    historial=None,
    k: int = TOP_K,
    min_relevance: float = MIN_RELEVANCE,
    verbose: bool = True,
) -> dict:
    historial = (historial or [])[-MAX_HISTORY_MESSAGES:]

    consulta = reformular_pregunta(pregunta, historial, llm)
    recuperados = recuperar(consulta, vector_store, k=k, min_relevance=min_relevance)
    recuperados = expandir_vecinos(recuperados, vector_store)
    docs = [d for d, _ in recuperados]

    if verbose:
        print(f"Consulta de búsqueda: {consulta}")
        for i, (d, s) in enumerate(recuperados, 1):
            print(f"  [{i}] {d.metadata.get('source')} Pág.{d.metadata.get('pagina')} ({s:.2f}): {d.page_content[:70]!r}")

    if not docs:
        return {
            "pregunta": pregunta,
            "pregunta_reformulada": consulta,
            "respuesta": NO_INFO,
            "sin_informacion": True,
            "fuentes": [],
            "fragmentos": [],
            "contextos": [],
        }

    mensajes = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=construir_mensaje_usuario(pregunta, docs, historial)),
    ]
    respuesta = llm.invoke(mensajes).content.strip()
    sin_info = respuesta.startswith(NO_INFO[:30])

    return {
        "pregunta": pregunta,
        "pregunta_reformulada": consulta,
        "respuesta": respuesta,
        "sin_informacion": sin_info,
        # Si el modelo se abstiene no se muestran fuentes (no respaldan ninguna respuesta).
        "fuentes": [] if sin_info else [_fuente_para_ui(d, s) for d, s in recuperados],
        "fragmentos": docs,
        "contextos": [d.page_content for d in docs],
    }

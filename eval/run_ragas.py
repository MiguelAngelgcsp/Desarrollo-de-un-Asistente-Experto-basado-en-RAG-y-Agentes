"""
Evaluación del RAG con Ragas (faithfulness, answer_relevancy, context_precision, context_recall).

Fase 1 (sin Ragas): genera la respuesta del RAG para cada pregunta y calcula métricas baratas:
    - hit_rate_recuperacion: ¿el fragmento esperado (documento+página) quedó entre los recuperados?
    - tasa de abstención correcta en las preguntas FUERA del corpus (control de alucinaciones)
    - tasa de abstención incorrecta en las preguntas DENTRO del corpus
Fase 2: Ragas (juez = LLM de Groq) sobre las preguntas dentro del corpus.

Ejemplos (cada configuración usa su propio índice, identificado por --tag):
    # Línea base (valores por defecto de config.py)
    python -m eval.run_ragas --tag base
    # Iteración de mejora: cambiar UN parámetro y volver a evaluar
    python -m eval.run_ragas --tag chunk1000 --chunk-size 1000 --chunk-overlap 100
    python -m eval.run_ragas --tag k5 --top-k 5
    # Comparar
    python -m eval.comparar base chunk1000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

RES_DIR = Path(__file__).resolve().parent / "resultados"
PREGUNTAS = Path(__file__).resolve().parent / "preguntas_eval.json"


# Modelo predeterminado de cada proveedor del juez (se cambia con --judge-model).
JUECES_POR_DEFECTO = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-3.5-flash-lite",   # requiere GOOGLE_API_KEY (.env) y: pip install langchain-google-genai
    "ollama": "qwen2.5:7b-instruct",     # requiere Ollama instalado y: pip install langchain-ollama
}


def crear_juez(a):
    """Devuelve el LLM (LangChain) que califica en Ragas, según --judge-provider.

    Usar un proveedor distinto al del RAG (Groq/gpt-oss) además evita que el modelo se califique a sí mismo.
    """
    proveedor = a.judge_provider
    modelo = a.judge_model or JUECES_POR_DEFECTO[proveedor]
    a.judge_model = modelo  # queda registrado en el resumen

    if proveedor == "groq":
        from src.llm import get_llm

        return get_llm(model=modelo, temperature=0.0)

    if proveedor == "gemini":
        clave = os.environ.get("GOOGLE_API_KEY")
        if not clave:
            raise RuntimeError("Falta GOOGLE_API_KEY en .env (créala gratis en https://aistudio.google.com/apikey).")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise RuntimeError("Falta el paquete: pip install langchain-google-genai") from e
        return ChatGoogleGenerativeAI(model=modelo, temperature=0.0, google_api_key=clave)

    if proveedor == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise RuntimeError("Falta el paquete: pip install langchain-ollama (y tener Ollama instalado)") from e
        return ChatOllama(model=modelo, temperature=0.0, base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))

    raise ValueError(f"Proveedor de juez no soportado: {proveedor}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="Nombre de la configuración (también nombra el índice)")
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--chunk-overlap", type=int)
    p.add_argument("--top-k", type=int)
    p.add_argument("--min-relevance", type=float)
    p.add_argument("--embedding-model")
    p.add_argument("--rebuild", action="store_true", help="Reconstruir el índice aunque exista")
    p.add_argument("--reusar-respuestas", action="store_true", help="No volver a llamar al RAG si ya hay respuestas guardadas")
    p.add_argument("--judge-provider", choices=sorted(JUECES_POR_DEFECTO), default="groq",
                   help="Proveedor del juez: groq (por defecto), gemini (API de Google) u ollama (modelo local)")
    p.add_argument("--judge-model", default=os.environ.get("RAGAS_JUDGE_MODEL"),
                   help="Modelo del juez. Si se omite se usa el predeterminado del proveedor (ver JUECES_POR_DEFECTO)")
    p.add_argument("--workers", type=int, default=2, help="Hilos de Ragas (bájalo a 1 si el proveedor responde 429)")
    p.add_argument("--pausa", type=float, default=2.0, help="Segundos entre preguntas en la Fase 1")
    p.add_argument("--limit", type=int, help="Evaluar solo las primeras N preguntas (para pruebas rápidas)")
    p.add_argument("--solo-fase1", action="store_true", help="Omitir Ragas")
    return p.parse_args()


def aplicar_config_en_entorno(a):
    """Debe ejecutarse ANTES de importar src.config."""
    os.environ["INDEX_TAG"] = a.tag
    for nombre, valor in (
        ("CHUNK_SIZE", a.chunk_size), ("CHUNK_OVERLAP", a.chunk_overlap), ("TOP_K", a.top_k),
        ("MIN_RELEVANCE", a.min_relevance), ("EMBEDDING_MODEL", a.embedding_model),
    ):
        if valor is not None:
            os.environ[nombre] = str(valor)


def con_reintentos(fn, intentos=4, espera=15):
    for i in range(intentos):
        try:
            return fn()
        except Exception as e:
            if i == intentos - 1:
                raise
            print(f"   ! {type(e).__name__}: reintentando en {espera * (i + 1)} s…")
            time.sleep(espera * (i + 1))


def fase1_generar(a, preguntas):
    from src import config
    from src.document_loader import load_documents
    from src.embeddings import get_embeddings
    from src.llm import get_llm
    from src.pipeline import rag_pipeline
    from src.splitter import split_documents
    from src.vectorstore import build_vector_store, indice_existe, load_vector_store

    emb = get_embeddings()
    if a.rebuild or not indice_existe():
        vs = build_vector_store(split_documents(load_documents()), emb, force_rebuild=True)
    else:
        vs = load_vector_store(emb)
    llm = get_llm()

    filas = []
    for i, q in enumerate(preguntas, 1):
        print(f"[{i}/{len(preguntas)}] {q['id']} {q['pregunta']}")
        r = con_reintentos(lambda: rag_pipeline(q["pregunta"], vs, llm, k=config.TOP_K, min_relevance=config.MIN_RELEVANCE))
        recuperados = [(d.metadata.get("source"), d.metadata.get("pagina")) for d in r["fragmentos"]]
        hit = None
        if q["en_corpus"]:
            esperados = {(f["documento"], p) for f in q["fuentes_esperadas"] for p in f["paginas"]}
            hit = any(rec in esperados for rec in recuperados)
        filas.append({
            "id": q["id"], "pregunta": q["pregunta"], "ground_truth": q["ground_truth"],
            "en_corpus": q["en_corpus"], "categoria": q["categoria"],
            "respuesta": r["respuesta"], "contextos": r["contextos"], "recuperados": recuperados,
            "sin_informacion": r["sin_informacion"], "hit_recuperacion": hit,
        })
        time.sleep(a.pausa)
    return filas


def metricas_basicas(filas):
    dentro = [f for f in filas if f["en_corpus"]]
    fuera = [f for f in filas if not f["en_corpus"]]
    return {
        "n_preguntas": len(filas), "n_dentro_corpus": len(dentro), "n_fuera_corpus": len(fuera),
        "hit_rate_recuperacion": round(sum(bool(f["hit_recuperacion"]) for f in dentro) / max(len(dentro), 1), 3),
        "abstencion_correcta_fuera_corpus": round(sum(f["sin_informacion"] for f in fuera) / max(len(fuera), 1), 3),
        "abstencion_incorrecta_dentro_corpus": round(sum(f["sin_informacion"] for f in dentro) / max(len(dentro), 1), 3),
    }


def fase2_ragas(a, filas):
    import pandas as pd
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from ragas.run_config import RunConfig

    from src.embeddings import get_embeddings
    from src.llm import get_llm

    dentro = [f for f in filas if f["en_corpus"]]
    ds = EvaluationDataset.from_list([
        {"user_input": f["pregunta"], "retrieved_contexts": f["contextos"] or [""],
         "response": f["respuesta"], "reference": f["ground_truth"]}
        for f in dentro
    ])

    juez = LangchainLLMWrapper(crear_juez(a))
    emb = LangchainEmbeddingsWrapper(get_embeddings())
    answer_relevancy.strictness = 1  # Groq no admite n>1 generaciones por llamada

    resultado = evaluate(
        ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=juez, embeddings=emb, raise_exceptions=False,
        run_config=RunConfig(timeout=240, max_retries=10, max_wait=60, max_workers=a.workers),
    )
    df = resultado.to_pandas()

    # Normaliza nombres de columnas entre versiones de Ragas
    mapa = {}
    for col in df.columns:
        c = col.lower()
        if "faithfulness" in c: mapa[col] = "faithfulness"
        elif "relevancy" in c or "relevance" in c: mapa[col] = "answer_relevancy"
        elif "context_precision" in c: mapa[col] = "context_precision"
        elif "context_recall" in c: mapa[col] = "context_recall"
    df = df.rename(columns=mapa)
    df.insert(0, "id", [f["id"] for f in dentro])
    df["categoria"] = [f["categoria"] for f in dentro]
    return df


def main():
    a = parse_args()
    aplicar_config_en_entorno(a)
    from src import config

    RES_DIR.mkdir(parents=True, exist_ok=True)
    preguntas = json.loads(PREGUNTAS.read_text(encoding="utf-8"))
    if a.limit:
        preguntas = preguntas[: a.limit]

    f_resp = RES_DIR / f"{a.tag}_respuestas.json"
    if a.reusar_respuestas and f_resp.exists():
        filas = json.loads(f_resp.read_text(encoding="utf-8"))
        print(f"Respuestas reutilizadas de {f_resp.name}")
    else:
        filas = fase1_generar(a, preguntas)
        f_resp.write_text(json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")

    resumen = {
        "tag": a.tag,
        "config": {"chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP, "top_k": config.TOP_K,
                   "min_relevance": config.MIN_RELEVANCE, "embedding_model": config.EMBEDDING_MODEL,
                   "llm": config.GROQ_MODEL, "juez_ragas": f"{a.judge_provider}:{a.judge_model}"},
        "basicas": metricas_basicas(filas),
    }

    if not a.solo_fase1:
        df = fase2_ragas(a, filas)
        df.to_csv(RES_DIR / f"{a.tag}_ragas_detalle.csv", index=False, encoding="utf-8")
        metricas = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        resumen["ragas"] = {m: (round(float(df[m].mean()), 3) if m in df else None) for m in metricas}
        resumen["ragas_nan"] = {m: int(df[m].isna().sum()) for m in metricas if m in df}
        if any(resumen["ragas_nan"].values()):
            print("\n⚠ Hay valores NaN (casi siempre límites de la API del juez). "
                  "Baja --workers o vuelve a correr con --reusar-respuestas.")

    (RES_DIR / f"{a.tag}_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== RESUMEN ===")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
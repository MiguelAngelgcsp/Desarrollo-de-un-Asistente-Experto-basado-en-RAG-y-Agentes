"""
Iteración solo de recuperación (NO llama a Groq).

Reutiliza las respuestas del baseline, recupera los contextos con la nueva configuración
(top_k / min_relevance) y evalúa con Ragas únicamente context_precision y context_recall
(juez Gemini). También calcula hit_rate_recuperacion.

NO mide faithfulness ni answer_relevancy: requieren regenerar las respuestas con el LLM de Groq.

Uso (desde la raíz del proyecto, con env_ragas activado):
    python -m eval.run_ctx_iter --tag k3 --top-k 3 --workers 1
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

RES_DIR = Path(__file__).resolve().parent / "resultados"
PREGUNTAS = Path(__file__).resolve().parent / "preguntas_eval.json"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="Nombre de la iteración (también nombra el índice)")
    p.add_argument("--base", default="base", help="Tag del baseline cuyas respuestas se reutilizan")
    p.add_argument("--top-k", type=int)
    p.add_argument("--min-relevance", type=float)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--judge-provider", default="gemini")
    p.add_argument("--judge-model", default=None)
    a = p.parse_args()

    # Debe hacerse ANTES de importar src.config
    os.environ["INDEX_TAG"] = a.tag
    if a.top_k is not None:
        os.environ["TOP_K"] = str(a.top_k)
    if a.min_relevance is not None:
        os.environ["MIN_RELEVANCE"] = str(a.min_relevance)

    from src import config
    from src.document_loader import load_documents
    from src.embeddings import get_embeddings
    from src.pipeline import expandir_vecinos, recuperar
    from src.splitter import split_documents
    from src.vectorstore import build_vector_store, indice_existe, load_vector_store

    from eval.run_ragas import crear_juez

    base = json.loads((RES_DIR / f"{a.base}_respuestas.json").read_text(encoding="utf-8"))
    preguntas = {q["id"]: q for q in json.loads(PREGUNTAS.read_text(encoding="utf-8"))}

    emb = get_embeddings()
    if not indice_existe():
        vs = build_vector_store(split_documents(load_documents()), emb, force_rebuild=True)
    else:
        vs = load_vector_store(emb)

    # --- Recuperación con la nueva configuración (sin LLM) ---
    filas = []
    for f in base:
        if not f["en_corpus"]:
            continue
        rec = recuperar(f["pregunta"], vs, k=config.TOP_K, min_relevance=config.MIN_RELEVANCE)
        rec = expandir_vecinos(rec, vs)
        docs = [d for d, _ in rec]
        recuperados = [(d.metadata.get("source"), d.metadata.get("pagina")) for d in docs]
        esperados = {(x["documento"], pg) for x in preguntas[f["id"]]["fuentes_esperadas"] for pg in x["paginas"]}
        filas.append({
            "id": f["id"], "pregunta": f["pregunta"], "ground_truth": f["ground_truth"],
            "respuesta": f["respuesta"], "categoria": f["categoria"],
            "contextos": [d.page_content for d in docs],
            "hit": any(r in esperados for r in recuperados),
        })
        print(f"{f['id']}: {len(docs)} contextos, hit={filas[-1]['hit']}")

    # --- Ragas: solo context_precision y context_recall ---
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import context_precision, context_recall
    from ragas.run_config import RunConfig

    ds = EvaluationDataset.from_list([
        {"user_input": f["pregunta"], "retrieved_contexts": f["contextos"] or [""],
         "response": f["respuesta"], "reference": f["ground_truth"]}
        for f in filas
    ])
    juez = LangchainLLMWrapper(crear_juez(SimpleNamespace(judge_provider=a.judge_provider, judge_model=a.judge_model)))
    resultado = evaluate(
        ds, metrics=[context_precision, context_recall],
        llm=juez, embeddings=LangchainEmbeddingsWrapper(emb), raise_exceptions=False,
        run_config=RunConfig(timeout=240, max_retries=10, max_wait=60, max_workers=a.workers),
    )
    df = resultado.to_pandas()
    mapa = {}
    for col in df.columns:
        c = col.lower()
        if "context_precision" in c:
            mapa[col] = "context_precision"
        elif "context_recall" in c:
            mapa[col] = "context_recall"
    df = df.rename(columns=mapa)
    df.insert(0, "id", [f["id"] for f in filas])
    df["hit_recuperacion"] = [f["hit"] for f in filas]
    df["n_contextos"] = [len(f["contextos"]) for f in filas]
    df.to_csv(RES_DIR / f"{a.tag}_ragas_detalle.csv", index=False, encoding="utf-8")

    resumen = {
        "tag": a.tag,
        "nota": "Iteración solo de recuperación: respuestas reutilizadas del baseline; "
                "faithfulness y answer_relevancy NO re-medidas (cuota diaria de Groq).",
        "config": {"chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP, "top_k": config.TOP_K,
                   "min_relevance": config.MIN_RELEVANCE, "embedding_model": config.EMBEDDING_MODEL,
                   "juez_ragas": f"{a.judge_provider}:{a.judge_model}"},
        "hit_rate_recuperacion": round(sum(f["hit"] for f in filas) / max(len(filas), 1), 3),
        "contextos_promedio": round(sum(len(f["contextos"]) for f in filas) / max(len(filas), 1), 1),
        "ragas": {m: (round(float(df[m].mean()), 3) if m in df else None) for m in ("context_precision", "context_recall")},
        "ragas_nan": {m: int(df[m].isna().sum()) for m in ("context_precision", "context_recall") if m in df},
    }
    (RES_DIR / f"{a.tag}_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== RESUMEN ===")
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())

"""
Paso 3 — Vectorización.
paraphrase-multilingual-MiniLM-L12-v2 (Sentence Transformers): multilingüe (el corpus mezcla
español e inglés y las preguntas son en español), liviano (~120M parámetros), corre en CPU
sin API externa y produce vectores de 384 dimensiones normalizados.
"""

from src.config import EMBEDDING_MODEL


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings  # import diferido (carga torch)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"[OK] Embeddings locales cargados: {EMBEDDING_MODEL}")
    return embeddings

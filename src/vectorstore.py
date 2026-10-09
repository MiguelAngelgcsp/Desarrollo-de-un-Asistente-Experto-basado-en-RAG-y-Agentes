"""
Paso 4 — Base vectorial (ChromaDB persistente, distancia coseno).
Cada vector guarda su texto y los metadatos (source, pagina, seccion, chunk_id)
que permiten citar la fuente de cada respuesta.
"""

import shutil

from langchain_chroma import Chroma

from src.config import COLLECTION_NAME, PERSIST_DIR


def indice_existe() -> bool:
    return PERSIST_DIR.exists() and any(PERSIST_DIR.iterdir())


def build_vector_store(chunks, embeddings_model, force_rebuild: bool = True):
    if force_rebuild and PERSIST_DIR.exists():
        shutil.rmtree(PERSIST_DIR)
        print(f"Índice anterior eliminado: {PERSIST_DIR}")
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Indexando {len(chunks)} fragmentos en ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=str(PERSIST_DIR),
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"[OK] Índice creado en {PERSIST_DIR} con {vector_store._collection.count()} fragmentos")
    return vector_store


def load_vector_store(embeddings_model):
    if not indice_existe():
        raise FileNotFoundError(
            f"No existe un índice en '{PERSIST_DIR}'. Ejecuta: python -m scripts.build_index"
        )
    vector_store = Chroma(
        persist_directory=str(PERSIST_DIR),
        embedding_function=embeddings_model,
        collection_name=COLLECTION_NAME,
    )
    print(f"Índice cargado: {vector_store._collection.count()} fragmentos ({PERSIST_DIR.name})")
    return vector_store

"""Construye (o reconstruye) el índice vectorial. No requiere API key.

Uso:  python -m scripts.build_index
"""

from src.document_loader import load_documents
from src.embeddings import get_embeddings
from src.splitter import split_documents
from src.vectorstore import build_vector_store


def main():
    chunks = split_documents(load_documents())
    build_vector_store(chunks, get_embeddings(), force_rebuild=True)


if __name__ == "__main__":
    main()

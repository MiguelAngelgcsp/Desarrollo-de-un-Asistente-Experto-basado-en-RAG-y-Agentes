"""Chat por consola (misma lógica que la web, con historial).

Uso:  python -m scripts.chat_cli
"""

from src.embeddings import get_embeddings
from src.llm import get_llm
from src.pipeline import rag_pipeline
from src.vectorstore import build_vector_store, indice_existe, load_vector_store
from src.document_loader import load_documents
from src.splitter import split_documents


def main():
    emb = get_embeddings()
    if indice_existe():
        vs = load_vector_store(emb)
    else:
        vs = build_vector_store(split_documents(load_documents()), emb)
    llm = get_llm()
    historial = []
    print("\nTutor Académico Konrad Lorenz. Escribe 'salir' para terminar.")
    while True:
        pregunta = input("\n❓ Tú: ").strip()
        if pregunta.lower() in ("salir", "exit", "quit", ""):
            break
        r = rag_pipeline(pregunta, vs, llm, historial=historial, verbose=True)
        print(f"\n🤖 {r['respuesta']}")
        historial += [{"role": "user", "content": pregunta}, {"role": "assistant", "content": r["respuesta"]}]


if __name__ == "__main__":
    main()

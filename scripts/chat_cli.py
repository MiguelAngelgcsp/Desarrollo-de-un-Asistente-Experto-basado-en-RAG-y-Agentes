"""Chat por consola con el AGENTE (misma lógica que la web, con historial y registro de tools).

Uso:  python -m scripts.chat_cli
"""

from src.agent import Agente
from src.document_loader import load_documents
from src.embeddings import get_embeddings
from src.llm import get_llm
from src.splitter import split_documents
from src.vectorstore import build_vector_store, indice_existe, load_vector_store


def main():
    emb = get_embeddings()
    vs = load_vector_store(emb) if indice_existe() else build_vector_store(split_documents(load_documents()), emb)
    agente = Agente(get_llm(), vs)
    historial = []
    print("\nTutorAE (agente). Escribe 'salir' para terminar.")
    while True:
        pregunta = input("\n❓ Tú: ").strip()
        if pregunta.lower() in ("salir", "exit", "quit", ""):
            break
        r = agente.responder(pregunta, historial)
        for t in r["tools_usadas"]:
            print(f"   🔧 {t['tool']}({t['args']})")
        print(f"\n🤖 {r['respuesta']}")
        print(f"   [{r['llamadas_llm']} llamadas al LLM · {r['tokens']} tokens · {r['segundos']} s]")
        historial += [{"role": "user", "content": pregunta}, {"role": "assistant", "content": r["respuesta"]}]


if __name__ == "__main__":
    main()

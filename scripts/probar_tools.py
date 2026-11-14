"""Prueba las tools SIN el LLM (con .invoke()) y muestra su salida. No necesita GROQ_API_KEY.

Uso:  python -m scripts.probar_tools
      python -m scripts.probar_tools "objetivos de la Fase A"      # otra consulta para buscar_en_base
(buscar_en_web solo devuelve resultados si TAVILY_API_KEY está definida.)
"""

import sys

from src.embeddings import get_embeddings
from src.tools import crear_tools
from src.vectorstore import build_vector_store, indice_existe, load_vector_store
from src.document_loader import load_documents
from src.splitter import split_documents


def main():
    consulta = " ".join(sys.argv[1:]) or "objetivos de la Fase Preliminar"
    emb = get_embeddings()
    vs = load_vector_store(emb) if indice_existe() else build_vector_store(split_documents(load_documents()), emb)
    tools = {t.name: t for t in crear_tools(vs)}

    print("\n=== Tools registradas ===")
    for t in tools.values():
        print(f"- {t.name}({', '.join(t.args)}): {t.description.splitlines()[0]}")

    print(f"\n=== buscar_en_base.invoke({{'consulta': {consulta!r}}}) ===")
    print(tools["buscar_en_base"].invoke({"consulta": consulta}))

    print("\n=== generar_referencia_apa.invoke({'documento': 'Fase Preliminar', 'paginas': '5-8'}) ===")
    print(tools["generar_referencia_apa"].invoke({"documento": "Fase Preliminar", "paginas": "5-8"}))

    print("\n=== listar_documentos_base.invoke({}) ===")
    print(tools["listar_documentos_base"].invoke({}))

    print("\n=== buscar_en_web.invoke({'consulta': 'TOGAF latest version'}) ===")
    print(tools["buscar_en_web"].invoke({"consulta": "TOGAF latest version"}))


if __name__ == "__main__":
    main()

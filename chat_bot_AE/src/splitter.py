"""
Paso 2 — Chunking.
RecursiveCharacterTextSplitter: corta primero por párrafos, luego por líneas y por
oraciones, y solo en último caso por palabras, para no partir ideas a la mitad.
Cada fragmento recibe metadatos extra para citar:
    chunk_id -> identificador único
    seccion  -> "Art. 43" (Reglamento) o "Sec. 5.5.3" (TOGAF), si se detecta
"""

import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE

_RE_ARTICULO = re.compile(r"ART[ÍI]CULO\s+(\d+)\s*[:.]")
_RE_SECCION = re.compile(r"^(\d+\.\d+(?:\.\d+)*)\s+[A-Z]", re.MULTILINE)


def _etiqueta(chunk_text: str, ultima: str | None):
    """Devuelve (etiqueta del fragmento, última etiqueta vista) heredando la anterior si no hay encabezado."""
    arts = _RE_ARTICULO.findall(chunk_text)
    if arts:
        etiqueta = f"Art. {arts[0]}" if len(set(arts)) == 1 else f"Arts. {arts[0]}-{arts[-1]}"
        return etiqueta, f"Art. {arts[-1]}"
    secs = _RE_SECCION.findall(chunk_text)
    if secs:
        return f"Sec. {secs[0]}", f"Sec. {secs[-1]}"
    return ultima, ultima


def split_documents(documents, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    ultima_por_fuente: dict[str, str | None] = {}
    contador: dict[str, int] = {}
    for chunk in chunks:
        fuente = chunk.metadata["source"]
        n = contador.get(fuente, 0)
        contador[fuente] = n + 1
        chunk.metadata["chunk_id"] = f"{fuente}::{n:04d}"

        etiqueta, ultima = _etiqueta(chunk.page_content, ultima_por_fuente.get(fuente))
        ultima_por_fuente[fuente] = ultima
        if etiqueta:
            chunk.metadata["seccion"] = etiqueta

    print(f"Documentos originales (páginas): {len(documents)}")
    print(f"Fragmentos generados:            {len(chunks)}  (size={chunk_size}, overlap={chunk_overlap})")
    return chunks

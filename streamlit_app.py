"""
Interfaz de chat en Streamlit (despliegue en Streamlit Community Cloud).

Avance 3: usa el AGENTE (src/agent.py), el mismo que app.py (Flask). El LLM decide qué tools llamar
(buscar_en_base, generar_referencia_apa, listar_documentos_base, buscar_en_web) y la interfaz muestra,
en cada respuesta, las tools usadas con sus argumentos y las fuentes separadas por origen (base / web).

Local:   streamlit run streamlit_app.py
Nube:    Streamlit Community Cloud (GROQ_API_KEY y TAVILY_API_KEY van en Settings -> Secrets, nunca en el repo)
"""

import os

# --- Chroma necesita SQLite >= 3.35; en algunas imágenes Linux es más viejo ---
try:
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

# Las claves de st.secrets pasan al entorno ANTES de importar src.* (que lee os.environ).
try:
    for _clave in ("GROQ_API_KEY", "TAVILY_API_KEY", "GROQ_MODEL", "TOP_K", "CHUNK_SIZE", "CHUNK_OVERLAP", "MIN_RELEVANCE"):
        if _clave in st.secrets:
            os.environ[_clave] = str(st.secrets[_clave])
except Exception:
    pass  # sin secrets.toml (ejecución local con .env)

from src.agent import Agente  # noqa: E402
from src.config import MAX_HISTORY_MESSAGES  # noqa: E402
from src.document_loader import load_documents  # noqa: E402
from src.embeddings import get_embeddings  # noqa: E402
from src.llm import get_llm  # noqa: E402
from src.splitter import split_documents  # noqa: E402
from src.vectorstore import build_vector_store, indice_existe, load_vector_store  # noqa: E402

EJEMPLOS = [
    "¿Cuáles son los objetivos principales de la Fase A (Architecture Vision)?",
    "¿Cuáles son las cuatro dimensiones que definen el alcance de una arquitectura?",
    "¿Qué es la Fase Preliminar de TOGAF?",
    "Hola, ¿qué puedes hacer?",
    "Dame la referencia APA de la Fase Preliminar, páginas 5 a 8",
    "¿Cuál es la versión más reciente de TOGAF publicada por The Open Group?",
]

st.set_page_config(page_title="TutorAE — Arquitectura Empresarial", page_icon="🎓", layout="centered")


@st.cache_resource(show_spinner="Cargando modelo e índice (solo la primera vez, ~1–2 min)…")
def iniciar_rag():
    """Se ejecuta UNA sola vez por servidor: embeddings + índice + LLM + agente."""
    embeddings = get_embeddings()
    if indice_existe():
        vector_store = load_vector_store(embeddings)
    else:  # en la nube el índice no viene en el repo (chroma/ está en .gitignore): se construye aquí
        vector_store = build_vector_store(split_documents(load_documents()), embeddings, force_rebuild=False)
    return Agente(get_llm(), vector_store)


def mostrar_tools(tools, meta):
    if not tools:
        st.caption("🔧 Sin tools: el agente respondió directamente.")
        return
    with st.expander(f"🔧 Tools usadas ({len(tools)})"):
        for t in tools:
            st.markdown(f"**{t['tool']}**")
            st.json(t["args"])
        if meta:
            st.caption(f"{meta['llamadas_llm']} llamadas al LLM · {meta['tokens']} tokens · {meta['segundos']} s")


def mostrar_web(resultados):
    if not resultados:
        return
    with st.expander(f"🌐 Fuentes web ({len(resultados)})"):
        for w in resultados:
            st.markdown(f"**[{w['titulo'] or w['url']}]({w['url']})** · `web`")
            st.caption(w["extracto"])
            st.divider()


def mostrar_fuentes(fuentes):
    if not fuentes:
        return
    with st.expander(f"📚 Fuentes de la base ({len(fuentes)})"):
        for f in fuentes:
            partes = [f"**{f['documento']}**"]
            if f.get("pagina"):
                partes.append(f"Pág. {f['pagina']}")
            if f.get("seccion"):
                partes.append(f["seccion"])
            partes.append(f"relevancia {round(f['relevancia'] * 100)}%")
            st.markdown(" · ".join(partes))
            st.caption(f["fragmento"])
            st.divider()


def mostrar_mensaje(m):
    with st.chat_message(m["role"]):
        if m.get("no_info"):
            st.warning(m["content"])
        else:
            st.markdown(m["content"])
        if m["role"] == "assistant":
            mostrar_tools(m.get("tools", []), m.get("meta"))
            mostrar_fuentes(m.get("fuentes"))
            mostrar_web(m.get("web"))


# ------------------------------------------------------------------ Estado
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# ------------------------------------------------------------------ Barra lateral
with st.sidebar:
    st.header("TutorAE")
    st.caption("Agente: consulta primero la base de TOGAF 9.1 y solo busca en la web (sitios oficiales) si la base no tiene la respuesta. Muestra las tools usadas y las fuentes por origen.")
    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()
    st.subheader("Prueba con:")
    for i, ejemplo in enumerate(EJEMPLOS):
        if st.button(ejemplo, key=f"ej{i}", use_container_width=True):
            st.session_state.pendiente = ejemplo

# ------------------------------------------------------------------ Cabecera
st.title("🎓 TutorAE")
st.caption("Agente tutor de Arquitectura Empresarial · TOGAF 9.1 (ADM, Fase Preliminar, Visión, Negocio e Información)")

try:
    agente = iniciar_rag()
except Exception as e:
    st.error(f"No se pudo iniciar el asistente: {e}")
    st.stop()

for m in st.session_state.mensajes:
    mostrar_mensaje(m)

if not st.session_state.mensajes:
    st.info("Hola. Haz una pregunta sobre TOGAF o usa un ejemplo de la barra lateral. "
            "Puedes hacer preguntas de seguimiento (por ejemplo: «¿y qué entradas necesita esa fase?»).")

# ------------------------------------------------------------------ Entrada
pregunta = st.chat_input("Escribe tu pregunta…") or st.session_state.pop("pendiente", None)

if pregunta:
    historial = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.mensajes[-MAX_HISTORY_MESSAGES:]
    ]
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    mostrar_mensaje(st.session_state.mensajes[-1])

    with st.chat_message("assistant"):
        with st.spinner("El agente está decidiendo qué tools usar…"):
            try:
                r = agente.responder(pregunta, historial)
            except Exception as e:
                st.error(f"No se pudo generar la respuesta ({type(e).__name__}). Intenta de nuevo en unos segundos.")
                st.session_state.mensajes.pop()  # no dejar la pregunta sin respuesta en el historial
                st.stop()
    st.session_state.mensajes.append(
        {"role": "assistant", "content": r["respuesta"], "fuentes": r["fuentes_base"], "web": r["fuentes_web"],
         "tools": r["tools_usadas"], "no_info": r["sin_informacion"],
         "meta": {"llamadas_llm": r["llamadas_llm"], "tokens": r["tokens"], "segundos": r["segundos"]}}
    )
    st.rerun()

"""
Paso 7 — LLM (Groq). La API key se lee del entorno (.env local o secrets de la plataforma).
"""

import os

from src.config import GROQ_MODEL, LLM_TEMPERATURE


def get_llm(model: str | None = None, temperature: float | None = None):
    from langchain_groq import ChatGroq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or api_key.startswith("gsk_PEGA"):
        raise RuntimeError(
            "Falta GROQ_API_KEY. Local: copia .env.example a .env y pega tu key. "
            "En la nube: defínela como variable de entorno / secret de la plataforma."
        )
    modelo = model or GROQ_MODEL
    llm = ChatGroq(
        model=modelo,
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
        api_key=api_key,
    )
    print(f"[OK] LLM configurado: {modelo} (Groq)")
    return llm

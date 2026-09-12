import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types



# CONFIGURACION
ASSISTANT_NAME = "TutorAE"
DOMAIN = "Arquitectura Empresarial (Ingeniería de Software II)"

MODEL_NAME = "gemini-3.6-flash"

TEMPERATURE = 0.2
MAX_TOKENS = 800

# "xml" -> <contexto>...</contexto> / "triple_quotes" -> """CONTEXTO ... """
DELIMITER_STYLE = "xml"
USE_FEW_SHOT = True

BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "knowledge_base" / "arquitectura_empresarial.md"



# CARGA DE VARIABLES DE ENTORNO (.env)


load_dotenv(BASE_DIR / ".env")



SYSTEM_PROMPT = f"""
Eres {ASSISTANT_NAME}, un tutor académico experto en {DOMAIN}.

# Rol
Ayudas a estudiantes a entender conceptos de la asignatura usando ÚNICAMENTE
el material que aparece en <contexto>. No eres un buscador general de
internet y no debes usar conocimiento externo al contexto entregado.

# Objetivo
- Explicar conceptos con claridad.
- Relacionar la pregunta con el material del contexto.
- Si el contexto no contiene información suficiente, indicarlo.
- No inventar información que no esté respaldada por el contexto.

# Reglas de comportamiento
1. Basa cada respuesta exclusivamente en el contenido de <contexto>.
2. Si el contexto no contiene información suficiente, dilo explícitamente en
   "respuesta" y usa "confianza": "baja".
3. Nunca inventes citas, artículos, normas o datos.
4. Usa un tono cercano y pedagógico.
5. Sé conciso y directo.

# Formato de salida
Responde SIEMPRE como JSON válido con esta estructura exacta:

{{
  "respuesta": "<explicación clara y directa>",
  "concepto_relacionado": "<concepto principal>",
  "fuente": "<sección del contexto en la que se basa la respuesta>",
  "confianza": "alta | media | baja"
}}

# Restricciones
- No agregues claves adicionales al JSON.
- No repitas el contexto completo.
- No respondas preguntas fuera del dominio de {DOMAIN}.
""".strip()



# FEW-SHOT PROMPTING


FEW_SHOT_EXAMPLES = [
    {
        "user": (
            "<contexto>\n"
            "TOGAF se organiza alrededor del ADM (Architecture Development "
            "Method), un ciclo iterativo de fases que guía el desarrollo de "
            "la arquitectura empresarial.\n"
            "</contexto>\n"
            "<pregunta>¿Qué es el ADM en TOGAF?</pregunta>"
        ),
        "assistant": (
            '{'
            '"respuesta":"El ADM es el método central de TOGAF: un ciclo '
            'iterativo de fases que guía el desarrollo de la arquitectura '
            'empresarial.",'
            '"concepto_relacionado":"TOGAF - ADM",'
            '"fuente":"Sección 2. TOGAF",'
            '"confianza":"alta"'
            '}'
        ),
    },
    {
        "user": (
            "<contexto>\n"
            "El BMM describe metas, objetivos, estrategias y tácticas de "
            "negocio.\n"
            "</contexto>\n"
            "<pregunta>¿Cuál es la diferencia entre ITIL y COBIT?</pregunta>"
        ),
        "assistant": (
            '{'
            '"respuesta":"El material del contexto no incluye información '
            'sobre ITIL ni COBIT, así que no puedo responder esta pregunta '
            'con base en el contexto entregado.",'
            '"concepto_relacionado":"No aplica",'
            '"fuente":"No encontrado en el contexto",'
            '"confianza":"baja"'
            '}'
        ),
    },
]



# CARGA DE LA BASE DE CONOCIMIENTO (contexto estático)


def load_knowledge_base(path: Path = KB_PATH) -> str:
    """
    Carga la base de conocimiento completa desde un archivo Markdown.

    En este avance el documento completo se usa como contexto estático:
    no hay chunking ni recuperación semántica (eso corresponde al avance
    de "RAG real").
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe la base de conocimiento: {path}")

    return path.read_text(encoding="utf-8").strip()


# ESTRATEGIAS DE DELIMITADORES


def build_context_block(context_text: str, style: str = DELIMITER_STYLE) -> str:
    """Envuelve el contexto usando la estrategia de delimitadores elegida."""
    if style == "xml":
        return f"<contexto>\n{context_text}\n</contexto>"

    if style == "triple_quotes":
        return f'"""CONTEXTO\n{context_text}\n"""'

    raise ValueError(f"Estilo de delimitador no soportado: {style}")


def build_question_block(question: str, style: str = DELIMITER_STYLE) -> str:
    """Envuelve la pregunta del estudiante con la misma estrategia."""
    question = question.strip()

    if style == "xml":
        return f"<pregunta>{question}</pregunta>"

    if style == "triple_quotes":
        return f'"""PREGUNTA\n{question}\n"""'

    raise ValueError(f"Estilo de delimitador no soportado: {style}")


def build_messages(question: str, context_text: str) -> list[dict]:
    """Construye los mensajes con few-shot y el contexto estático."""
    messages = []

    if USE_FEW_SHOT:
        for example in FEW_SHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "parts": [{"text": example["user"]}],
            })
            messages.append({
                "role": "model",
                "parts": [{"text": example["assistant"]}],
            })

    context_block = build_context_block(context_text)
    question_block = build_question_block(question)

    messages.append({
        "role": "user",
        "parts": [{"text": f"{context_block}\n{question_block}"}],
    })

    return messages


# CLIENTE Y GENERACIÓN CON GEMINI


def get_gemini_client() -> genai.Client:
    """Crea el cliente oficial de Gemini usando GEMINI_API_KEY (.env)."""
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "No se encontró GEMINI_API_KEY.\n"
            "Crea un archivo .env en la raíz del proyecto (puedes copiar "
            ".env.example) con la línea:\n\n"
            "  GEMINI_API_KEY=tu_clave_aqui\n"
        )

    return genai.Client(api_key=api_key)


def call_gemini(client: genai.Client, messages: list[dict]) -> str:
    """Genera la respuesta final usando Gemini."""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=messages,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_TOKENS,
            response_mime_type="application/json",
        ),
    )

    return response.text



# FLUJO DEL ASISTENTE (Avance 1: sin recuperación semántica)


def ask(question: str) -> dict:
    """
    Ejecuta el flujo de este avance:

    base de conocimiento completa -> contexto estático delimitado
    -> few-shot + system prompt -> Gemini -> respuesta JSON.
    """
    client = get_gemini_client()
    context_text = load_knowledge_base()

    messages = build_messages(question, context_text)
    raw_output = call_gemini(client, messages)

    try:
        result = json.loads(raw_output)
    except json.JSONDecodeError:
        result = {"respuesta_cruda": raw_output}

    return result



# EJECUCIÓN


def main() -> None:
    print(f"=== {ASSISTANT_NAME} | Avance 1 (Prompt Design) ===")
    print(f"Modelo: {MODEL_NAME}")
    print(f"Delimitador: {DELIMITER_STYLE}")
    print(f"Few-shot activo: {USE_FEW_SHOT}")

    if len(sys.argv) > 1:
        preguntas = [" ".join(sys.argv[1:])]
    else:
        preguntas = [
            "¿Qué es el ADM en TOGAF?",
            "¿Qué relación hay entre metas, objetivos y tácticas en el BMM?",
            "¿Qué es el C4ISR y por qué es importante?",
            "¿Cuánto cuesta implementar SAP en una empresa mediana?",
        ]

    for pregunta in preguntas:
        print(f"\nPregunta: {pregunta}")

        try:
            resultado = ask(pregunta)
            print("--- Respuesta ---")
            print(json.dumps(resultado, ensure_ascii=False, indent=2))

        except Exception as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main()

# Desarrollo-de-un-Asistente-Experto-basado-en-RAG-y-Agentes
# Asistente Experto basado en RAG y Agentes — Avance 1

## Enfoque elegido
**Tutor Académico Personalizado**, sobre la asignatura de Arquitectura
Empresarial (Ingeniería de Software II). El material de referencia es
`knowledge_base/arquitectura_empresarial.md`.

## Alcance de este avance

Según el enunciado, Avance 1 corresponde únicamente a **diseño de prompts**:

- **System Prompt**: define el rol del tutor, sus reglas de comportamiento
  y el formato de salida obligatorio (JSON).
- **Few-Shot Prompting**: dos ejemplos (`FEW_SHOT_EXAMPLES`) que muestran al
  modelo cómo responder cuando sí hay información en el contexto y cómo
  responder cuando no la hay.
- **Estrategias de delimitadores**: el contexto y la pregunta se envuelven
  con tags XML (`<contexto>...</contexto>`, `<pregunta>...</pregunta>`) o,
  alternativamente, con triple comillas (`"""CONTEXTO ... """`), configurable
  con `DELIMITER_STYLE`.

**Este avance no implementa recuperación semántica (RAG real).** Toda la
base de conocimiento se inyecta completa como contexto estático en el
prompt. El chunking, los embeddings y la búsqueda por similitud coseno
quedan para el siguiente avance, cuando se aborde el RAG propiamente dicho.

## Flujo actual

```text
Base de conocimiento (.md, completa)
        ↓
Delimitadores (XML o triple comillas)
        ↓
System Prompt + Few-Shot Examples + Contexto + Pregunta
        ↓
Gemini 3.6 Flash
        ↓
Respuesta en JSON
```

## Sobre el uso de la API de Gemini (nota importante)

El enunciado general del proyecto pide un sistema que funcione **de manera
local, preservando la privacidad de los datos**. En esta etapa del curso
todavía no se ha visto cómo desplegar y ejecutar un modelo de lenguaje de
forma local (por ejemplo con Ollama o llama.cpp), por lo que este avance usa
la API en la nube de Gemini para poder cumplir con los demás requisitos
(system prompt, few-shot, delimitadores, formato de salida). Se documenta
esta salvedad de forma explícita: la migración a un modelo local se
contempla para cuando ese contenido se cubra en el curso.

## Estructura

```text
asistente-tutor-ae/
├── main.py
├── requirements.txt
├── .env
├── .gitignore
├── README.md
└── knowledge_base/
    └── arquitectura_empresarial.md
```

## Instalación

```cmd
pip install -r requirements.txt
```

## Configurar la clave de Gemini (.env)

1. Crear `.env`:
2. Abre `.env` y coloca tu clave real:

   ```env
   GEMINI_API_KEY=tu_clave_aqui
   ```

`.env` está incluido en `.gitignore`, por lo que la clave nunca se sube al
repositorio.

## Ejecución

```cmd
python main.py
```

También puedes enviar una pregunta directamente:

```cmd
python main.py "¿Qué es el BMM?"
```

## ¿Dónde está cada pieza del diseño de prompts?

- `SYSTEM_PROMPT` → instrucciones del sistema (rol, reglas, formato JSON).
- `FEW_SHOT_EXAMPLES` → ejemplos few-shot.
- `build_context_block()` / `build_question_block()` → estrategias de
  delimitadores (XML / triple comillas).
- `build_messages()` → arma la conversación completa que se envía a Gemini.
- `call_gemini()` → llamada al modelo generativo.
- `ask()` → coordina el flujo de este avance.

## Próximos avances

- Implementar el RAG real: chunking de la base de conocimiento, generación
  de embeddings, y recuperación por similitud coseno (o vectorial) antes de
  construir el contexto.
- Evaluar la viabilidad de un modelo local para cumplir con el requisito de
  privacidad del enunciado.

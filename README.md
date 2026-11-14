# TutorAE — Agente de Arquitectura Empresarial (TOGAF 9.1) · Avance 3

Agente basado en **RAG + tool calling**. Decide **cuándo** consultar la base de conocimientos y **qué herramienta** usar:
busca en la base (TOGAF 9.1), genera referencias APA exactas de los documentos y, solo si la base no tiene la respuesta
y la pregunta es de información pública o actual, busca en la web en una lista fija de sitios oficiales. **Cita** cada
afirmación (`Fuente: documento, Pág.` para la base, `[web: URL]` para la web) y **se abstiene** si no hay información.

- **URL pública:** <https://tutor-ae.streamlit.app/> *(redesplegar con este código y configurar los secrets, ver §6)*
- **Repositorio:** <https://github.com/MiguelAngelgcsp/Desarrollo-de-un-Asistente-Experto-basado-en-RAG-y-Agentes>
- **Integrantes:** Miguel Angel Correa Muñoz · Miguel Angel Gomez Cruz
- **Curso:** Desarrollo de Aplicaciones con IA — Fundación Universitaria Konrad Lorenz, Bogotá D.C., 2026-2 (Avance 3)

El Avance 3 reutiliza del Avance 2 la ingesta, el chunking, el índice vectorial, el conjunto de evaluación de Ragas y la
interfaz. Cambia cómo se conecta la búsqueda con el LLM: el pipeline fijo (`src/pipeline.py`, se conserva como línea base
de comparación) pasa a ser una **tool** que el LLM decide llamar (`src/agent.py`).

## Arquitectura del agente

![Arquitectura del agente](docs/diagrama_agente.png)

Flujo: pregunta → **LLM** (Groq, con las tools enlazadas) → ¿tool call? → el **programa** ejecuta la tool → resultado al LLM →
… → respuesta con citas. Un saludo se responde sin tools.

| Qué sale del sistema | Hacia | Contenido |
|---|---|---|
| Prompt del LLM | Groq | System Prompt, historial (≤ 8 mensajes), pregunta y **solo los fragmentos** que devolvió `buscar_en_base` (≤ 5 × 700 caracteres por llamada). Nunca el corpus completo |
| Consulta web | Tavily | **Solo el texto de la consulta** (≤ 200 caracteres) y la lista de dominios permitidos |
| **No sale** | — | PDF del corpus, índice ChromaDB, claves, historial hacia la web |

### Tools

| Tool | Tipo | Qué hace / cuándo | Argumentos | Límites (en código) |
|---|---|---|---|---|
| `buscar_en_base` | búsqueda | Recuperación del Avance 2 (similitud coseno + umbral). Para toda pregunta sobre TOGAF; una llamada por tema | `consulta` | `src/tools.py`: máx. **5 fragmentos** por llamada, **700 caracteres** por fragmento, consulta ≤ 300 car.; cada fragmento lleva documento, página y sección |
| `generar_referencia_apa` | tarea del dominio | Referencia y cita en texto APA 7 de un documento de la base, con datos exactos del catálogo (no de memoria) | `documento`, `paginas` (opcional, p. ej. `5-8`) | Solo documentos del catálogo; si no hay coincidencia única devuelve error con la lista |
| `listar_documentos_base` | tarea (consulta exacta) | Lista los documentos de la base | — | — |
| `buscar_en_web` | web | Búsqueda en sitios oficiales permitidos (ver política) | `consulta` | máx. **3 resultados**, extracto ≤ **300 car.**, consulta ≤ 200 car., dominios fijos, rechazada si antes no se llamó `buscar_en_base` |

Cada tool tiene docstring que dice qué hace y cuándo usarla (es lo que ve el LLM). Los resultados de la base se entregan
dentro de `<fragmento origen="base" documento=… pagina=…>` y los de la web en `<resultado origen="web" url=…>`; lo que
hay dentro es **dato, no instrucción**.

**Prueba de la tool sin LLM** (no necesita `GROQ_API_KEY`):

```bash
python -m scripts.probar_tools                  # .invoke() de las cuatro tools y su salida
python -m scripts.probar_tools "Phase A objectives"
```

### Bucle del agente (`src/agent.py`)

- Implementación **manual** con `llm.bind_tools(tools)` de LangChain (Groq, `openai/gpt-oss`).
- **Límites en código:** `MAX_ITERACIONES = 5` llamadas al LLM con tools por pregunta y `MAX_TOOL_CALLS_PREGUNTA = 6`. Si se
  alcanza el límite, una última llamada **sin tools** pide responder con lo recopilado (o `NO_INFO`).
- Un saludo o charla casual se responde sin tools (instrucción del prompt; verificado en el caso C1 de la evaluación).
- El agente puede llamar `buscar_en_base` varias veces (pregunta con dos temas, o reformular si el primer resultado es pobre).
- Se eliminó la llamada de reformulación del Avance 2: el propio agente escribe una consulta autocontenida al llamar la tool.
- Errores de tool (argumentos inválidos, tool inexistente, fallo de red) se devuelven al LLM como `ToolMessage` sin romper el bucle.

### Registro de uso de tools

Cada pregunta agrega una línea JSON a `logs/uso_tools.jsonl` (ignorado por git):

```json
{"timestamp": "2026-11-13T23:46:12+00:00", "pregunta": "...", "tools": [{"tool": "buscar_en_base",
 "args": {"consulta": "preliminary phase objectives"}, "n_resultados": 5, "resultado": "ok", "segundos": 0.019}],
 "llamadas_llm": 2, "tokens": 1873, "segundos": 2.41, "limite_iteraciones": false, "sin_informacion": false}
```

### Prompt del sistema (`SYSTEM_PROMPT_AGENTE` en `src/prompts.py`)

Cambios frente al Avance 2: **ya no lleva el contexto** (llega como resultado de las tools); incluye una sección `<tools>` que
dice cuándo usar cada una; responde solo con lo que devuelvan las tools; cita `(Fuente: documento, Pág. n)` o `[web: URL]` y
separa la respuesta por origen si usó ambas; trata como datos lo que devuelvan las tools y lo que escriba el usuario
(instrucciones maliciosas); mantiene la respuesta exacta `No encontré información sobre esto en la base de conocimientos.`;
los ejemplos Few-Shot ahora muestran la acción (tool) además de la respuesta. El prompt del Avance 2 (`SYSTEM_PROMPT`) se
conserva para el pipeline fijo.

## Política de búsqueda web

- **Cuándo:** solo si `buscar_en_base` no tiene la respuesta **y** la pregunta trata de información pública o actual (versión
  reciente, norma vigente, definición general). La regla está en el docstring de la tool, en el System Prompt y **se aplica en
  código**: el bucle rechaza `buscar_en_web` si en esa pregunta aún no se llamó `buscar_en_base`.
- **Sitios permitidos** (`WEB_DOMINIOS_PERMITIDOS` en `src/tools.py`, fijos; el LLM no puede cambiarlos; además se descartan
  resultados fuera de la lista):

| Sitio | Por qué es confiable para el dominio |
|---|---|
| `opengroup.org` (incl. `pubs.opengroup.org`) | The Open Group es el autor y custodio oficial de TOGAF |
| `omg.org` | Object Management Group: estándares BPMN y BMM, usados junto con TOGAF |
| `iso.org` | Normas como ISO/IEC/IEEE 42010 (descripción de arquitecturas) |
| `ieee.org` | Estándares y literatura revisada de arquitectura de software y empresarial |
| `sei.cmu.edu` | Software Engineering Institute (Carnegie Mellon): referencia académica de arquitectura |

- **Resultados acotados:** máx. 3; cada uno devuelve solo título, URL y extracto corto, marcado `origen="web"`.
- **Respuesta separada por origen:** base → `(Fuente: documento, Pág. n)`; web → `[web: URL]`. La interfaz los muestra en secciones distintas.
- **Privacidad:** a Tavily solo sale la consulta; el corpus nunca. Sin `TAVILY_API_KEY` la tool responde «no disponible» y el agente sigue funcionando solo con la base.
- Si no hay respuesta ni en la base ni en los sitios permitidos, el agente responde `NO_INFO` y la interfaz muestra un aviso sin fuentes.

## Instalación, ejecución y despliegue (Avance 3)

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env     # completa GROQ_API_KEY y TAVILY_API_KEY (Tavily: https://app.tavily.com, plan gratuito)
python -m scripts.build_index           # opcional: la app crea el índice si no existe
python -m scripts.probar_tools          # prueba de las tools sin LLM
streamlit run streamlit_app.py          # http://localhost:8501   (alternativas: python app.py · python -m scripts.chat_cli)
```

**Despliegue (Streamlit Community Cloud):** igual que en §6; en *Secrets* añade además `TAVILY_API_KEY`
(ver `.streamlit/secrets.toml.example`). Las claves nunca van en el repositorio (`.gitignore` excluye `.env` y `secrets.toml`).
En Docker/Render/Hugging Face define `GROQ_API_KEY` y `TAVILY_API_KEY` como variables de entorno.

**Interfaz:** historial de conversación; en cada respuesta, desplegable **🔧 Tools usadas** (nombre y argumentos, llamadas al LLM,
tokens y tiempo), **📚 Fuentes de la base** (documento, página, sección, fragmento) y **🌐 Fuentes web** (título, URL, extracto).

## Evaluación del agente

Script: `eval/run_agente.py` (usa el mismo modelo, índice y juez que el Avance 2; compara contra `eval/resultados/base_*`).

```bash
python -m eval.run_agente --casos                                        # 8 casos de comportamiento
python -m eval.run_agente --ragas --judge-provider gemini --workers 1    # Ragas del agente + tabla vs Avance 2
python -m eval.run_agente --costo                                        # pipeline fijo vs agente (llamadas, tokens, tiempo)
python -m eval.run_agente --todo --judge-provider gemini --workers 1
```

- **Ragas:** las mismas 21 preguntas del Avance 2 (16 dentro del corpus); `retrieved_contexts` = fragmentos que devolvieron las tools.
  Salidas en `eval/resultados/`: `agente_resumen.json`, `agente_ragas_detalle.csv`, `comparacion_agente.md`.
- **Casos de comportamiento** (`eval/casos_comportamiento.json`, el comportamiento esperado está escrito antes de ejecutar):
  saludo, pregunta compuesta, 2 tareas del dominio (APA), información pública/actual (web), pregunta que responde la base
  (sin web), instrucción maliciosa y pregunta fuera de la base y de la web. Cada caso comprueba automáticamente tools usadas,
  argumentos, orden, límites de longitud y URLs permitidas; salida en `agente_casos.md|json`.
- **Costo:** llamadas al LLM, tokens y tiempo promedio por pregunta, pipeline fijo vs agente (`agente_costo.md`).

**Resultados:** *pendientes de ejecutar con las claves reales* (requieren `GROQ_API_KEY`, `TAVILY_API_KEY` y `GOOGLE_API_KEY`).
Pega aquí las tablas generadas en `eval/resultados/` y úsalas en el informe PDF.

| Métrica | Avance 2 (pipeline fijo) | Avance 3 (agente) |
|---|---|---|
| Faithfulness | 0.689 | _pendiente_ |
| Answer relevancy | 0.563 | _pendiente_ |
| Context precision | 0.178 | _pendiente_ |
| Context recall | 0.500 | _pendiente_ |

> Nota: el agente ya no expande los fragmentos vecinos (n±1) como el pipeline del Avance 2, para respetar el tope de 5
> fragmentos por llamada; puede compensarlo reformulando o llamando la tool de nuevo.

---

# Base RAG (Avance 2)

## 1. Dominio y corpus

Dominio: **tutor de Arquitectura Empresarial** para estudiantes de Ingeniería de Software II (Fundación
Universitaria Konrad Lorenz).

| Documento (`corpus/`) | Páginas | Fragmentos | Contenido |
|---|---|---|---|
| `TOGAF_9.1_Cap5_ADM.pdf` | 12 | 73 | Cap. 5: ciclo ADM, alcance e integración |
| `TOGAF_V9.1-FasePreliminar.pdf` | 12 | 54 | Fase Preliminar |
| `TOGAF_V9.1-VisionArquitectura.pdf` | 10 | 51 | Fase A: Visión de Arquitectura |
| `TOGAF_V9.1-ArquitecturaNegocio.pdf` | 14 | 74 | Fase B: Arquitectura de Negocio |
| `TOGAF_V9.1-ArquitecturaInformacion.pdf` | 12 | 51 | Fase C: Arquitecturas de Sistemas de Información |
| **Total** | **60** | **303** | |

Los cinco documentos cubren de forma secuencial el ADM, que es el contenido que el estudiante debe dominar.
Al ser un corpus coherente y acotado, se pueden formular preguntas "cercanas" fuera del corpus (Fase G, TOGAF 10,
Scrum…) para medir si el sistema se abstiene en lugar de inventar.

## 2. Flujo RAG implementado

![Diagrama del flujo RAG](docs/diagrama_flujo_rag.png)

| Etapa | Archivo | Decisión | Justificación |
|---|---|---|---|
| Ingesta | `src/document_loader.py` | PDF (PyMuPDF), DOCX, TXT, HTML; limpieza de marcas de agua y encabezados | PyMuPDF conserva mejor espacios y orden de lectura; quitar texto repetido evita que domine la similitud |
| Chunking | `src/splitter.py` | `RecursiveCharacterTextSplitter`, **600 caracteres, solape 100**, cortes `\n\n` → `\n` → `. ` → ` ` | Un fragmento por idea, con definiciones completas; el solape evita perder frases en el borde |
| Metadatos | `src/splitter.py` | `source`, `pagina`, `seccion` (Sec. 5.x), `chunk_id` | Permiten citar y localizar los fragmentos vecinos |
| Vectorización | `src/embeddings.py` | `paraphrase-multilingual-MiniLM-L12-v2` (384 dim, normalizado) | Multilingüe (preguntas en español, TOGAF en inglés), liviano, local en CPU, sin API |
| Índice | `src/vectorstore.py` | ChromaDB persistente (`chroma/<tag>`), distancia coseno | Persistencia y una carpeta por configuración experimental |
| Recuperación | `src/pipeline.py` | Similitud coseno, **top_k = 5**, **umbral de relevancia 0.20** | k pequeño reduce ruido; el umbral corta antes del LLM si nada es relevante |
| Expansión de vecinos | `src/pipeline.py` | Se agregan los fragmentos n−1 y n+1 de cada resultado | Las listas de TOGAF (objetivos, entradas, entregables) quedan repartidas en varios fragmentos; los vecinos aportan el contexto contiguo (mitiga el problema, pero no garantiza listas completas: ver §8) |
| Reformulación | `src/pipeline.py` | El LLM reescribe la pregunta de seguimiento como autocontenida | "¿Y qué entradas necesita esa fase?" no sirve como consulta vectorial; reformulada sí |
| Generación | `src/prompts.py`, `src/llm.py` | Groq (`GROQ_MODEL`, por defecto `openai/gpt-oss-20b`; despliegue y evaluación final: `openai/gpt-oss-120b`), temperatura 0 | Respuestas deterministas y trazables |

> Usa el **mismo modelo** (`GROQ_MODEL`) en la evaluación Ragas, el despliegue y el informe.

### Prompt (refinamiento del Avance 1) — `src/prompts.py`

- **System Prompt:** rol (TutorAE), objetivo, 9 reglas y formato de salida.
- **Delimitadores XML:** `<contexto><fragmento id fuente pagina seccion>`, `<historial>`, `<pregunta>`; lo que está dentro es dato, no instrucción.
- **Few-Shot:** 3 ejemplos (respuesta con cita, pregunta de seguimiento, caso fuera del contexto).
- **Formato de salida:** respuesta directa + viñetas si hay varios elementos + cita `(Fuente: archivo, Pág. n)`.

### Control de alucinaciones (dos capas)

1. **Umbral de relevancia:** si ningún fragmento lo supera, responde *"No encontré información sobre esto en la base de conocimientos."* sin llamar al LLM.
2. **Regla del System Prompt:** si hay fragmentos pero no contienen la respuesta, el LLM debe emitir exactamente esa frase; la interfaz no muestra fuentes.

## 3. Estructura

```
chat_bot_AE/
├── streamlit_app.py          # Interfaz desplegada (Streamlit Community Cloud) — usa el agente
├── src/tools.py              # [Av.3] tools: buscar_en_base, generar_referencia_apa, listar_documentos_base, buscar_en_web
├── src/agent.py              # [Av.3] bucle del agente + registro de uso (logs/uso_tools.jsonl)
├── eval/run_agente.py        # [Av.3] Ragas del agente, costo y casos de comportamiento
├── eval/casos_comportamiento.json
├── scripts/probar_tools.py   # [Av.3] prueba de las tools sin LLM
├── app.py                    # Interfaz alternativa local (Flask)
├── requirements.txt          # runtime (torch CPU, langchain, chroma, streamlit…)
├── requirements-eval.txt     # + Ragas, pandas, matplotlib
├── .env.example              # plantilla de variables (la key real NO se sube)
├── .streamlit/secrets.toml.example
├── Dockerfile                # alternativa de despliegue en plataformas con Docker
├── corpus/                   # documentos a vectorizar (5 PDF de TOGAF)
├── src/                      # config, document_loader, splitter, embeddings, vectorstore, prompts, llm, pipeline
├── scripts/                  # build_index.py, chat_cli.py
├── eval/
│   ├── preguntas_eval.json   # 21 preguntas con ground truth (5 fuera del corpus)
│   ├── run_ragas.py          # evaluación (Ragas + hit-rate + abstención)
│   ├── calibrar_umbral.py    # ayuda para elegir MIN_RELEVANCE
│   ├── comparar.py           # tabla + gráfico antes/después
│   ├── run_ctx_iter.py       # evaluación sin tokens de context precision y reacall
│   └── resultados/           # salidas de cada evaluación
├── docs/                     # diagrama, generar_diagrama.py, texto del informe
└── templates/ static/        # interfaz Flask
```

## 4. Instalación y ejecución local

Requisitos: **Python 3.11 o 3.12** (3.14 aún no tiene wheels para varias librerías) y una API key gratuita de [Groq](https://console.groq.com).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env        # Windows: copy .env.example .env   → pega tu GROQ_API_KEY en .env

python -m scripts.build_index     # (opcional) crea el índice; la app lo crea sola si no existe
streamlit run streamlit_app.py    # abre http://localhost:8501
# alternativas:  python app.py (Flask, http://localhost:5000)  ·  python -m scripts.chat_cli (consola)
```

La primera ejecución descarga el modelo de embeddings (~470 MB).

## 5. Evaluación con Ragas

Conjunto: `eval/preguntas_eval.json` — **21 preguntas**: 16 dentro del corpus (con *ground truth* y página esperada)
y 5 fuera del corpus (2 sin relación alguna y 3 "cercanas" al dominio).

### Instalación (Python 3.12, entorno aparte)

```powershell
py -3.12 -m venv env_ragas
.\env_ragas\Scripts\Activate.ps1
pip install -r requirements-eval.txt
pip check
```

En `.env` se requieren `GROQ_API_KEY` (genera las respuestas del RAG) y `GOOGLE_API_KEY` (juez de Ragas con Gemini). Se usa Google como juez porque Groq limita los modelos `gpt-oss` a 200.000 tokens diarios y la evaluación completa agota ese cupo.

### Ejecución

```powershell
python -m eval.calibrar_umbral                                                      # opcional: revisar MIN_RELEVANCE
python -m eval.run_ragas --tag base --workers 1 --judge-provider gemini             # línea base
python -m eval.run_ragas --tag base --reusar-respuestas --workers 1 --judge-provider gemini   # repetir solo Ragas
python -m eval.run_ragas --tag <iteracion> --<parametro> <valor> --workers 1 --judge-provider gemini
python -m eval.comparar base <iteracion>
```

Métricas (16 preguntas dentro del corpus; juez = LLM de Groq): **faithfulness, answer_relevancy, context_precision,
context_recall**. Además, sin costo de LLM: **hit-rate de recuperación**, **abstención correcta fuera del corpus**
(control de alucinaciones) y **abstención incorrecta dentro del corpus**; ayudan a atribuir el problema a
chunking, recuperación o generación.

### Si algo falla

| Error | Solución |
|---|---|
| `No module named 'langchain_community.chat_models.vertexai'` | Parchear `env_ragas\Lib\site-packages\ragas\llms\base.py` (línea ~12): envolver el import de `ChatVertexAI` en `try/except ImportError` y asignar `ChatVertexAI = None` si falla. Se pierde al reinstalar `ragas` |
| `cannot import name 'ModelProfile'` | Paquetes `langchain-*` desalineados: `pip install -U langchain-core langchain-huggingface langchain-groq langchain-google-genai` y `pip check` |
| `429 ... tokens per day` (Groq) | Cuota diaria agotada. Esperar el reinicio y usar `--reusar-respuestas --workers 1` |
| `Falta GOOGLE_API_KEY` | Crearla en https://aistudio.google.com/apikey y agregarla al `.env` |
| Valores `NaN` en `ragas_nan` | Límite del juez. Repetir con `--reusar-respuestas --workers 1` antes de reportar |

### Juez de Ragas (proveedor y modelo)

Por defecto el juez es un modelo de Groq (`llama-3.3-70b-versatile`). Se puede cambiar sin tocar el código:

```bash
python -m eval.run_ragas --tag base --judge-model llama-3.1-8b-instant             # otro modelo de Groq
python -m eval.run_ragas --tag base --judge-provider gemini                       # Gemini (requiere GOOGLE_API_KEY en .env)
python -m eval.run_ragas --tag base --judge-provider ollama --judge-model qwen2.5:7b-instruct   # modelo local con Ollama
```

Para Gemini: `pip install langchain-google-genai`; para Ollama: instalar Ollama y `pip install langchain-ollama`.
El juez usado queda registrado en `eval/resultados/<tag>_resumen.json` (campo `juez_ragas`). **Usa el mismo juez en la
línea base y en la iteración**, o la comparación no es válida.

Si el proveedor devuelve 429 (límite de tokens, frecuente en Groq gratuito porque Ragas hace varias llamadas por
pregunta), las métricas quedan vacías (NaN) en vez de detener la corrida: baja `--workers` a 1, prueba con `--limit 3`
o reparte la evaluación en varios días.

### Iteración de mejora (cambia UN parámetro y compara)

```bash
python -m eval.run_ragas --tag k3 --top-k 3                                 # ejemplo: top_k 5 → 3 este comando es cuando se usa el modelo de groq por defecto y se quiere validar el retriver

python -m eval.run_ctx_iter --tag k3 --top-k 3 --workers 1 #Este solo evalua el context presision y el recall
python -m eval.run_ragas --tag chunk450 --chunk-size 450 --chunk-overlap 80   # ejemplo: chunking
python -m eval.comparar base k10                                              # tabla + gráfico en eval/resultados/
```

Guía de lectura: *context_precision* baja → ruido recuperado (bajar `top_k`, subir umbral); *context_recall* baja →
no se recupera lo necesario (subir `top_k`, cambiar chunking); *faithfulness* baja → el LLM agrega datos que no
están en el contexto (endurecer el prompt); *answer_relevancy* baja → respuestas largas o fuera de foco.

Nota: el modelo de embeddings procesa ~128 tokens; 600 caracteres equivalen a ~130–150 tokens, así que está en el
límite. Si el hit-rate es bajo, probar `--chunk-size 450` es una iteración natural.

Si Groq devuelve 429 (límite), usa `--workers 1` y `--reusar-respuestas`.

### Resultados

**Línea base (`--tag base`).** Valores por defecto de `src/config.py`: chunk_size 600, overlap 100, top_k 5,
umbral de relevancia 0.20, embeddings `paraphrase-multilingual-MiniLM-L12-v2`, LLM `openai/gpt-oss-120b` (Groq).
El conjunto completo (`eval/preguntas_eval.json`) tiene 21 preguntas (16 dentro del corpus y 5 fuera); la línea base
reportada en el informe de entrega se calculó sobre 16 preguntas (11 dentro del corpus y 5 fuera). Las cuatro métricas
de Ragas se calculan sobre las preguntas que sí están en el corpus.

| Métrica | Línea base | Qué mide |
|---|---|---|
| Faithfulness | 0.689 | La respuesta se apoya en los fragmentos recuperados |
| Answer relevancy | 0.563 | La respuesta contesta lo preguntado |
| Context precision | 0.178 | Los fragmentos recuperados son útiles y están bien ordenados |
| Context recall | 0.500 | Los fragmentos contienen la información de la referencia |

**Interpretación.**
- La métrica más baja es **context precision (0.178)**: según la guía de lectura de arriba, apunta a la etapa de
  **recuperación** (ruido en el contexto). Una causa probable es la expansión de vecinos: con top_k = 5 y los fragmentos
  n−1 y n+1 de cada resultado, la interfaz llega a mostrar unas 14–15 fuentes por respuesta, lo que diluye la precisión.
- **Context recall (0.500)** indica que cerca de la mitad de la información de referencia no llega al contexto: se
  atribuye al **chunking y la recuperación** (listas repartidas entre fragmentos y páginas; ver §8).
- Faithfulness (0.689) y answer relevancy (0.563) son intermedias; pueden verse afectadas por el contexto
  ruidoso o incompleto que recibe la **generación**.
- Para mejorar precision: reducir el ruido (menos fragmentos o vecinos solo para los mejores resultados, subir
  `MIN_RELEVANCE`). Para mejorar recall: dividir el texto sin cortar en el salto de página y completar las listas con
  los fragmentos contiguos.

**Iteración de mejora (antes/después).** *Pendiente de ejecutar.* Elegir un solo parámetro (por ejemplo `--top-k 3` o
`--min-relevance`, dado que la métrica más baja es context precision), correr `python -m eval.run_ragas --tag <nombre> ...`
con el mismo juez y luego `python -m eval.comparar base <nombre>`.

| Métrica | Antes (`base`) | Después (`iteración`) | Δ |
|---|---|---|---|
| Faithfulness | 0.689 | no re-media|-|
| Answer relevancy | 0.563 | no re-media|-|
| Context precision | 0.178 |0.107 |-0.071|
| Context recall | 0.500 |0.281 |-0.219|

## 6. Despliegue en la nube (Streamlit Community Cloud)

1. Sube el proyecto a GitHub (sin `.env`; el `.gitignore` ya lo excluye).
2. Entra a <https://share.streamlit.io> → **Continue with GitHub** → **Create app** → *Deploy a public app from GitHub*.
3. Repositorio y rama del proyecto; **Main file path:** `streamlit_app.py`; elige la URL de la app.
4. **Advanced settings:** *Python version* **3.12**; en **Secrets** pega (ver `.streamlit/secrets.toml.example`):
   ```toml
   GROQ_API_KEY = "gsk_..."
   GROQ_MODEL = "openai/gpt-oss-120b"
   TAVILY_API_KEY = "tvly-..."   # Avance 3: búsqueda web
   ```
5. **Deploy.** El primer arranque (5–10 min) instala dependencias, descarga el modelo y construye el índice; después
   solo lo carga. La URL pública de este proyecto es <https://tutor-ae.streamlit.app/>.
6. Las apps sin visitas por 12 horas se duermen; se reactivan con un clic.

Alternativa: el `Dockerfile` sirve para plataformas con Docker (Render, Railway, Fly.io…); requiere variable
`GROQ_API_KEY` y suficiente RAM (≥ 1 GB) para PyTorch y el modelo de embeddings.

### Seguridad

- La API key solo vive en `.env` (ignorado por git) o en los *secrets* de la plataforma. **Nunca** en el repositorio.
- `.gitignore` excluye `.env`, `chroma/` y `.streamlit/secrets.toml`. Si una key se expuso alguna vez, genera una nueva en Groq y revoca la anterior.

## 7. Interfaz conversacional

- El historial se guarda en la sesión (últimos 8 mensajes se usan en cada turno) → admite preguntas de seguimiento.
- Cada respuesta muestra un desplegable **Fuentes** con documento, página, sección, relevancia y fragmento.
- Si la información no está en el corpus, se muestra un aviso amarillo y no se listan fuentes.
- Barra lateral: ejemplos de preguntas y botón **Nueva conversación**.

**Pruebas manuales en la app desplegada** (<https://tutor-ae.streamlit.app/>):

| Prueba | Resultado |
|---|---|
| «¿Cuáles son los objetivos principales de la Fase A (Architecture Vision)?» | Correcta: dos objetivos de la sección 7.1 (Pág. 2), con cita de archivo, página y sección |
| Seguimiento «¿Y qué entradas necesita esa fase?» | Entiende que «esa fase» es la Fase A, pero lista pasos de la sección 7.4 en lugar de las entradas de 7.3 (incompleta) |
| Seguimiento «¿Y qué entregables produce?» | Cubre 5 de los ~9 elementos de la sección 7.5 e incluye un paso (definir el alcance) que no es un entregable |
| Fuera del corpus: «¿Cuáles son las actividades y entregables de la Fase G (Implementation Governance) del ADM?» | Correcto: «No encontré información sobre esto en la base de conocimientos.», sin fuentes («Implementation Governance» no aparece en ningún PDF del corpus) |

## 8. Limitaciones conocidas

- **Listas largas y preguntas de seguimiento cortas:** en las pruebas, «entradas» y «entregables» de la Fase A salieron
  incompletos o mezclados con pasos de otra sección (ver §7). Causa probable: el texto se divide página por página
  (un fragmento no cruza el salto de página) y la consulta reformulada recupera la sección equivocada. Mejoras
  propuestas: dividir cada PDF como un texto continuo, completar las listas con los fragmentos contiguos y ajustar `top_k`.
- **Evaluación y límites de la API:** Ragas hace varias llamadas al LLM juez por pregunta; con los límites gratuitos de
  Groq las métricas pueden quedar vacías (ver «Juez de Ragas»).

- Los puntajes de Ragas dependen del LLM juez y varían entre corridas; compara siempre con el mismo juez y modelo.
- TOGAF está en inglés y se consulta en español: la similitud multilingüe es menor que la monolingüe.
- El índice no se actualiza solo: si cambias `corpus/`, ejecuta `python -m scripts.build_index` (en Streamlit Cloud, reinicia la app).
- Streamlit Community Cloud ofrece memoria limitada (~2.7 GB compartidos); si la app se reinicia por memoria, usar un modelo de embeddings más liviano.

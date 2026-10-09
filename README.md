# TutorAE — Asistente RAG de Arquitectura Empresarial (TOGAF 9.1)

Asistente experto basado en **RAG** con memoria conversacional. Responde preguntas sobre el ADM de TOGAF 9.1
(Fase Preliminar, Visión de Arquitectura, Arquitectura de Negocio y de Información), **cita la fuente de cada
respuesta** (documento, página, sección y fragmento) y **se abstiene** cuando la información no está en la base de
conocimientos.

- **URL pública:** `PENDIENTE — pega aquí la URL de Streamlit Community Cloud (https://<nombre>.streamlit.app)`
- **Repositorio:** `PENDIENTE — pega aquí la URL de GitHub`

---

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
| Expansión de vecinos | `src/pipeline.py` | Se agregan los fragmentos n−1 y n+1 de cada resultado | Las listas de TOGAF (objetivos, entradas, entregables) quedan repartidas en varios fragmentos |
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
├── streamlit_app.py          # Interfaz desplegada (Streamlit Community Cloud)
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

```bash
pip install -r requirements-eval.txt

python -m eval.calibrar_umbral                  # opcional: revisar MIN_RELEVANCE
python -m eval.run_ragas --tag base             # línea base (configuración de src/config.py)
```

Métricas (16 preguntas dentro del corpus; juez = LLM de Groq): **faithfulness, answer_relevancy, context_precision,
context_recall**. Además, sin costo de LLM: **hit-rate de recuperación**, **abstención correcta fuera del corpus**
(control de alucinaciones) y **abstención incorrecta dentro del corpus**; ayudan a atribuir el problema a
chunking, recuperación o generación.

### Iteración de mejora (cambia UN parámetro y compara)

```bash
python -m eval.run_ragas --tag k10 --top-k 10                                 # ejemplo: top_k 5 → 10
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

> Completar con `eval/resultados/comparacion.md` después de correr la línea base y la iteración.

| Métrica | Antes (`base`) | Después (`...`) | Δ |
|---|---|---|---|
| Faithfulness | | | |
| Answer relevancy | | | |
| Context precision | | | |
| Context recall | | | |

## 6. Despliegue en la nube (Streamlit Community Cloud)

1. Sube el proyecto a GitHub (sin `.env`; el `.gitignore` ya lo excluye).
2. Entra a <https://share.streamlit.io> → **Continue with GitHub** → **Create app** → *Deploy a public app from GitHub*.
3. Repositorio y rama del proyecto; **Main file path:** `streamlit_app.py`; elige la URL de la app.
4. **Advanced settings:** *Python version* **3.12**; en **Secrets** pega (ver `.streamlit/secrets.toml.example`):
   ```toml
   GROQ_API_KEY = "gsk_..."
   GROQ_MODEL = "openai/gpt-oss-120b"
   ```
5. **Deploy.** El primer arranque (5–10 min) instala dependencias, descarga el modelo y construye el índice; después
   solo lo carga. La URL pública queda en `https://<nombre>.streamlit.app`.
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

## 8. Limitaciones conocidas

- Los puntajes de Ragas dependen del LLM juez y varían entre corridas; compara siempre con el mismo juez y modelo.
- TOGAF está en inglés y se consulta en español: la similitud multilingüe es menor que la monolingüe.
- El índice no se actualiza solo: si cambias `corpus/`, ejecuta `python -m scripts.build_index` (en Streamlit Cloud, reinicia la app).
- Streamlit Community Cloud ofrece memoria limitada (~2.7 GB compartidos); si la app se reinicia por memoria, usar un modelo de embeddings más liviano.

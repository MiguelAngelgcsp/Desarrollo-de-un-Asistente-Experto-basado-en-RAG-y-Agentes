# 1. Flujo RAG implementado y justificación de parámetros

## 1.1 Diagrama

*(Insertar `diagrama_flujo_rag.png`)*

**Cómo se lee.** El sistema tiene dos recorridos. En la **indexación** (se ejecuta una vez) los PDF se limpian, se
fragmentan, se convierten en vectores y se guardan en ChromaDB junto con sus metadatos. En la **consulta** (en
cada mensaje) la pregunta se reformula con el historial, se buscan los fragmentos más parecidos, se descartan los
poco relevantes, se agregan los fragmentos vecinos y el LLM redacta la respuesta citando la fuente. Si ningún
fragmento supera el umbral, el sistema responde que no tiene información sin llamar al LLM.

## 1.2 Corpus

| Documento | Páginas | Fragmentos | Contenido |
|---|---|---|---|
| TOGAF_9.1_Cap5_ADM.pdf | 12 | 73 | Ciclo ADM, alcance e integración |
| TOGAF_V9.1-FasePreliminar.pdf | 12 | 54 | Fase Preliminar |
| TOGAF_V9.1-VisionArquitectura.pdf | 10 | 51 | Fase A: Visión de Arquitectura |
| TOGAF_V9.1-ArquitecturaNegocio.pdf | 14 | 74 | Fase B: Arquitectura de Negocio |
| TOGAF_V9.1-ArquitecturaInformacion.pdf | 12 | 51 | Fase C: Sistemas de Información |
| **Total** | **60** | **303** | |

**Pertinencia.** El asistente es un tutor de Arquitectura Empresarial para Ingeniería de Software II. Los cinco
documentos cubren de forma secuencial el ADM de TOGAF (de la fase preliminar a la fase C), que es el contenido que el
estudiante debe dominar. Al ser un corpus coherente y acotado, la evaluación puede incluir preguntas "cercanas" fuera
del corpus (por ejemplo ITIL, COBIT o TOGAF 10) para medir si el sistema se abstiene en lugar de inventar.

## 1.3 Parámetros y justificación

| Etapa | Parámetro elegido | Justificación |
|---|---|---|
| Ingesta | PyMuPDF; se eliminan marca de agua ("Evaluation Copy"), derechos de autor y encabezados repetidos | PyMuPDF conserva mejor los espacios y el orden de lectura que pypdf. Quitar el texto repetido evita que domine la similitud |
| Chunking | `RecursiveCharacterTextSplitter`, **600 caracteres**, solape **100** (≈17 %), separadores `\n\n` → `\n` → `. ` → ` ` | Los fragmentos son lo bastante pequeños para que cada uno trate una sola idea y lo bastante grandes para conservar una definición completa. El solape evita perder frases en los bordes. Tamaño real: mediana 545, máximo 599 caracteres |
| Metadatos | `source`, `pagina`, `seccion` (por ejemplo "Sec. 5.5.3"), `chunk_id` | Permiten citar documento, página y sección, y localizar los fragmentos vecinos. 300 de 303 fragmentos tienen sección |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2`, 384 dimensiones, normalizados | Multilingüe: las preguntas están en español y TOGAF en inglés. Liviano, corre en CPU, sin costo ni API externa |
| Índice | ChromaDB persistente, distancia coseno, una carpeta por configuración (`chroma/<tag>`) | Persistencia en disco y facilidad para comparar configuraciones en la evaluación antes/después |
| Recuperación | Similitud coseno, **top_k = 5**, umbral de relevancia **0.20** | k pequeño reduce el ruido en el contexto; el umbral corta antes del LLM cuando nada es relevante |
| Expansión de vecinos | Se agregan los fragmentos n−1 y n+1 de cada resultado | Las listas de TOGAF (objetivos, entradas, entregables) suelen quedar repartidas en varios fragmentos; los vecinos recuperan la lista completa |
| Reformulación | El LLM reescribe la pregunta de seguimiento como autocontenida | "¿Y la siguiente fase?" no sirve como consulta vectorial; reformulada sí |
| Generación | Groq, temperatura 0, system prompt + few-shot + delimitadores XML | Respuestas deterministas y trazables; los XML separan datos de instrucciones |

## 1.4 Prompt (refinado del Avance 1)

- **System prompt:** rol (TutorAE), objetivo, 9 reglas y formato de salida.
- **Reglas clave:** responder solo con el contexto; frase exacta de abstención; el historial solo sirve para entender la pregunta; citar `(Fuente: archivo, Pág. n)`; no inventar fases, normas ni autores.
- **Delimitadores XML:** `<contexto>`, `<fragmento>`, `<historial>`, `<pregunta>`. Todo lo que va dentro se trata como dato, no como instrucción.
- **Few-shot:** 3 ejemplos (respuesta con cita, pregunta de seguimiento y caso fuera del contexto).

## 1.5 Control de alucinaciones (dos capas)

1. **Umbral de relevancia:** si ningún fragmento lo supera, se responde "No encontré información sobre esto en la base de conocimientos." sin llamar al LLM.
2. **Regla del prompt:** si hay fragmentos pero no contienen la respuesta, el LLM debe emitir exactamente esa frase y la interfaz no muestra fuentes.

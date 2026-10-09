# Guía del PDF de entrega (Avance 2)

Exporta este contenido a PDF (Word/Docs → PDF) con las imágenes pegadas. Sigue el orden de la rúbrica.

## 1. Portada
Título, integrantes, curso, fecha, **URL pública** y **enlace al repositorio**.

## 2. Diagrama del flujo RAG y parámetros
- Imagen: `docs/diagrama_flujo_rag.png` (se regenera con `python docs/generar_diagrama.py`).
- Texto listo para pegar: `docs/seccion1_diagrama_y_parametros.md`.

## 3. Capturas de consultas sobre el RAG + análisis
Pega 4–5 capturas (consola `python -m scripts.chat_cli` o la web) y, para cada una, 2–3 líneas:

| Consulta | Qué se recuperó (doc/pág.) | ¿La respuesta es correcta y está citada? | Observación |
|---|---|---|---|
| ¿Cuáles son los objetivos principales de la Fase A (Architecture Vision)? | | | |
| ¿Cuáles son las cuatro dimensiones del alcance de una arquitectura? | | | |
| ¿Qué es la Fase Preliminar de TOGAF? | | | |
| Una pregunta fuera del corpus (p. ej. Fase G o TOGAF 10) | | | |

## 4. Resultados Ragas
1. Tabla de la línea base (`eval/resultados/base_resumen.json`): 4 métricas Ragas + hit-rate + abstenciones.
2. Gráfico: `eval/resultados/comparacion.png`.
3. **Análisis** (las tres preguntas de la rúbrica):
   - ¿Qué métrica salió más baja?
   - ¿A qué componente se atribuye (chunking, recuperación o generación)? Apóyate en el hit-rate y en las filas con peor puntaje de `*_ragas_detalle.csv`.
   - ¿Qué se haría para mejorarla?
4. **Iteración de mejora**: parámetro modificado, motivo, tabla antes/después (`comparacion.md`) e interpretación.
5. Control de alucinaciones: abstención correcta en las 5 preguntas fuera del corpus (y cuáles fallaron).

## 5. Interfaz desplegada
Capturas de la URL pública (con la barra de direcciones visible):
1. Pantalla inicial.
2. Conversación con **seguimiento** y el desplegable **Fuentes** abierto:
   «¿Cuáles son los objetivos principales de la Fase A?» → «¿Y qué entradas necesita esa fase?» → «¿Y qué entregables produce?»
3. Caso **fuera del corpus** (aviso amarillo), en una conversación nueva. Usa una pregunta que NO esté en los
   few-shot del prompt (el ejemplo de ITIL vs COBIT ya está ahí): p. ej. «¿Cuáles son las actividades de la Fase G (Implementation Governance)?» o «¿Qué cambios introdujo TOGAF 10 frente a TOGAF 9.1?».

## 6. Enlaces finales
URL pública · URL del repositorio GitHub.

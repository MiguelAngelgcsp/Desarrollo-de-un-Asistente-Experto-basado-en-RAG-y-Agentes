"""
Avance 3 — Tools del agente.

Tools (LangChain `@tool`, con docstring que le dice al LLM QUÉ hace y CUÁNDO usarla):

    buscar_en_base(consulta)              búsqueda     recuperación del Avance 2 convertida en tool
    generar_referencia_apa(documento, …)  tarea        referencia y cita APA 7 de un documento de la base
    listar_documentos_base()              tarea        consulta exacta del catálogo de documentos
    buscar_en_web(consulta)               web          búsqueda controlada (Tavily), solo "primero la base"

Los LÍMITES están escritos aquí, en código (constantes de abajo), no en el prompt: el LLM no puede cambiarlos.
Cada tool devuelve (content, artifact): `content` es el texto que ve el LLM y `artifact` son los datos
estructurados que usan la interfaz, el log y la evaluación (el LLM nunca ve el artifact).

La base de conocimientos NUNCA sale del sistema: al LLM solo viajan los fragmentos que devuelve
`buscar_en_base` (máx. MAX_FRAGMENTOS × MAX_CHARS_FRAGMENTO caracteres por llamada). Al buscador web solo viaja
el texto de la consulta (máx. MAX_CHARS_CONSULTA_WEB caracteres).
"""

from __future__ import annotations

import os
import re
import unicodedata
from urllib.parse import urlparse

import requests
from langchain_core.tools import tool

from src.config import MIN_RELEVANCE, TOP_K
from src.pipeline import recuperar
from src.prompts import _escapar

# ----------------------------------------------------------------------------------------------
# LÍMITES (en código, no en el prompt)
# ----------------------------------------------------------------------------------------------
MAX_FRAGMENTOS = min(TOP_K, 5)       # máx. fragmentos que devuelve buscar_en_base por llamada
MAX_CHARS_FRAGMENTO = 700            # máx. caracteres de cada fragmento (el chunk mide 600)
MAX_CHARS_CONSULTA = 300             # máx. caracteres de la consulta a la base
MAX_CHARS_CONSULTA_WEB = 200         # máx. caracteres que salen hacia el buscador web
WEB_MAX_RESULTADOS = 3               # máx. resultados web
WEB_MAX_CHARS_EXTRACTO = 300         # máx. caracteres del extracto de cada resultado web
WEB_TIMEOUT_S = 15

# Sitios permitidos para la búsqueda web: fijados aquí, el LLM no puede cambiarlos.
WEB_DOMINIOS_PERMITIDOS = (
    "opengroup.org",       # The Open Group: autor oficial de TOGAF (pubs.opengroup.org = documentación oficial)
    "omg.org",             # Object Management Group: BPMN, BMM (modelos usados con TOGAF)
    "iso.org",             # ISO/IEC/IEEE 42010 (descripción de arquitecturas) y otras normas
    "ieee.org",            # IEEE: estándares y literatura de arquitectura empresarial
    "sei.cmu.edu",         # Software Engineering Institute (CMU): referencia de arquitectura de software
)

TAVILY_URL = "https://api.tavily.com/search"

NO_RESULTADOS_BASE = (
    "SIN_RESULTADOS: ningún fragmento de la base de conocimientos supera el umbral de relevancia "
    "para esta consulta."
)

# ----------------------------------------------------------------------------------------------
# Catálogo de documentos (datos exactos para citar; el LLM no debe inventarlos de memoria)
# ----------------------------------------------------------------------------------------------
_BASE_URL_TOGAF = "https://pubs.opengroup.org/architecture/togaf9-doc/arch/"
_TOGAF = "TOGAF® Version 9.1"

CATALOGO_DOCUMENTOS: dict[str, dict] = {
    "TOGAF_9.1_Cap5_ADM.pdf": {
        "titulo": "Introduction to the ADM",
        "capitulo": 5,
        "descripcion": "Cap. 5: ciclo ADM, adaptación del ADM, alcance e integración de la arquitectura",
        "url": _BASE_URL_TOGAF + "chap05.html",
        "alias": ["adm", "cap5", "capitulo 5", "ciclo adm", "introduccion"],
    },
    "TOGAF_V9.1-FasePreliminar.pdf": {
        "titulo": "Preliminary Phase",
        "capitulo": 6,
        "descripcion": "Cap. 6: Fase Preliminar",
        "url": _BASE_URL_TOGAF + "chap06.html",
        "alias": ["preliminar", "fase preliminar", "preliminary", "cap6", "capitulo 6"],
    },
    "TOGAF_V9.1-VisionArquitectura.pdf": {
        "titulo": "Phase A: Architecture Vision",
        "capitulo": 7,
        "descripcion": "Cap. 7: Fase A, Visión de Arquitectura",
        "url": _BASE_URL_TOGAF + "chap07.html",
        "alias": ["vision", "fase a", "architecture vision", "cap7", "capitulo 7"],
    },
    "TOGAF_V9.1-ArquitecturaNegocio.pdf": {
        "titulo": "Phase B: Business Architecture",
        "capitulo": 8,
        "descripcion": "Cap. 8: Fase B, Arquitectura de Negocio",
        "url": _BASE_URL_TOGAF + "chap08.html",
        "alias": ["negocio", "fase b", "business architecture", "cap8", "capitulo 8"],
    },
    "TOGAF_V9.1-ArquitecturaInformacion.pdf": {
        "titulo": "Phase C: Information Systems Architectures — Data Architecture",
        "capitulo": 10,
        "descripcion": "Cap. 10: Fase C, Arquitecturas de Sistemas de Información (datos)",
        "url": _BASE_URL_TOGAF + "chap10.html",
        "alias": ["informacion", "fase c", "datos", "data architecture", "information systems", "cap10", "capitulo 10"],
    },
}
AUTOR = "The Open Group"
ANIO = 2011


def _norm(texto: str) -> str:
    texto = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def _resolver_documento(consulta: str) -> str | None:
    """Devuelve el nombre exacto del archivo que corresponde a `consulta`, o None si no hay coincidencia clara."""
    q = _norm(consulta).strip()
    for archivo in CATALOGO_DOCUMENTOS:
        if q in (_norm(archivo), _norm(archivo.rsplit(".", 1)[0])):
            return archivo
    aciertos = []
    for archivo, d in CATALOGO_DOCUMENTOS.items():
        claves = [_norm(a) for a in d["alias"]] + [_norm(d["titulo"]), _norm(archivo.rsplit(".", 1)[0])]
        if any(c and (c in q or q in c) for c in claves if len(c) > 2 and len(q) > 2):
            aciertos.append(archivo)
    return aciertos[0] if len(aciertos) == 1 else None


def _formatear_paginas(paginas: str | None) -> tuple[str, str]:
    """'5' -> ('p. 5', '5'); '5-8' / '5 a 8' -> ('pp. 5–8', '5–8'). Devuelve ('', '') si no hay páginas válidas."""
    if not paginas:
        return "", ""
    m = re.fullmatch(r"\s*(\d{1,4})\s*(?:[-–—]|a|al|hasta)?\s*(\d{1,4})?\s*", str(paginas).lower())
    if not m:
        return "", ""
    ini, fin = m.group(1), m.group(2)
    if fin and int(fin) > int(ini):
        return f"pp. {ini}–{fin}", f"{ini}–{fin}"
    return f"p. {ini}", ini


# ----------------------------------------------------------------------------------------------
# Fábrica de tools (necesita el vector store ya cargado)
# ----------------------------------------------------------------------------------------------
def crear_tools(vector_store):
    """Devuelve [buscar_en_base, generar_referencia_apa, listar_documentos_base, buscar_en_web]."""

    @tool(response_format="content_and_artifact")
    def buscar_en_base(consulta: str):
        """Busca en la base de conocimientos (documentos TOGAF 9.1: ADM, Fase Preliminar, Visión de
        Arquitectura, Arquitectura de Negocio y de Información) y devuelve fragmentos con documento y página.

        Úsala para CUALQUIER pregunta sobre esos temas, antes que cualquier otra tool. Si la pregunta
        trata varios temas, llámala una vez por tema con una consulta corta y específica (en español o
        inglés técnico, p. ej. "Phase A objectives"). NO la uses en saludos ni en charla casual.
        Devuelve como máximo 5 fragmentos; si no hay nada relevante devuelve SIN_RESULTADOS.

        Args:
            consulta: texto a buscar; autocontenido (sin pronombres como "eso" o "esa fase").
        """
        consulta = " ".join(str(consulta).split())[:MAX_CHARS_CONSULTA]
        if not consulta:
            return "ERROR: consulta vacía.", []
        pares = recuperar(consulta, vector_store, k=MAX_FRAGMENTOS, min_relevance=MIN_RELEVANCE)[:MAX_FRAGMENTOS]
        if not pares:
            return NO_RESULTADOS_BASE, []

        partes, artifact = [], []
        for i, (doc, score) in enumerate(pares, 1):
            texto = " ".join(doc.page_content.split())
            if len(texto) > MAX_CHARS_FRAGMENTO:
                texto = texto[:MAX_CHARS_FRAGMENTO].rstrip() + "…"
            documento = os.path.basename(doc.metadata.get("source", "?"))
            pagina = doc.metadata.get("pagina")
            seccion = doc.metadata.get("seccion")
            attrs = f'id="{i}" origen="base" documento="{documento}" pagina="{pagina if pagina else "N/A"}"'
            if seccion:
                attrs += f' seccion="{seccion}"'
            partes.append(f"<fragmento {attrs}>\n{_escapar(texto)}\n</fragmento>")
            artifact.append({
                "origen": "base", "documento": documento, "pagina": pagina, "seccion": seccion,
                "fragmento": texto, "relevancia": round(float(score), 3),
            })
        return "\n".join(partes), artifact

    @tool(response_format="content_and_artifact")
    def generar_referencia_apa(documento: str, paginas: str | None = None):
        """Genera la referencia bibliográfica en formato APA 7 y la cita en texto de un documento de la
        base de conocimientos, con datos exactos tomados del catálogo (autor, año, título, URL).

        Úsala cuando el usuario pida citar, referenciar o dar la "referencia APA" de uno de los
        documentos de la base. No inventes los datos bibliográficos: llama a esta tool. Si el usuario
        no dice qué documento, pregúntale o usa listar_documentos_base.

        Args:
            documento: documento o tema que pide el usuario (p. ej. "Fase Preliminar", "Architecture Vision",
                "Arquitectura de Negocio", "capítulo 5" o el nombre del archivo PDF).
            paginas: páginas citadas, solo si el usuario las dio (p. ej. "5" o "5-8"). Si no, omítelo.
        """
        archivo = _resolver_documento(str(documento))
        if archivo is None:
            disponibles = "; ".join(f"{a} ({d['descripcion']})" for a, d in CATALOGO_DOCUMENTOS.items())
            return (f"ERROR: no se encontró un documento único para «{documento}». "
                    f"Documentos disponibles: {disponibles}."), []
        d = CATALOGO_DOCUMENTOS[archivo]
        pag_ref, pag_cita = _formatear_paginas(paginas)
        referencia = (f"{AUTOR}. ({ANIO}). {d['titulo']}. In *{_TOGAF}* (chap. {d['capitulo']}"
                      f"{', ' + pag_ref if pag_ref else ''}). {d['url']}")
        cita = f"({AUTOR}, {ANIO}{', ' + pag_ref if pag_ref else ''})"
        contenido = (f"documento: {archivo}\nreferencia_apa: {referencia}\ncita_en_texto: {cita}\n"
                     "(Los datos provienen del catálogo del sistema; la referencia usa la edición TOGAF 9.1 de 2011.)")
        return contenido, [{"origen": "catalogo", "documento": archivo, "referencia_apa": referencia,
                            "cita_en_texto": cita}]

    @tool(response_format="content_and_artifact")
    def listar_documentos_base():
        """Lista los documentos que contiene la base de conocimientos (nombre de archivo, capítulo y tema).

        Úsala cuando el usuario pregunte qué documentos o temas cubre la base, o cuando necesites el
        nombre exacto de un documento para citarlo.
        """
        filas = [f"- {a}: {d['descripcion']}" for a, d in CATALOGO_DOCUMENTOS.items()]
        return "Documentos de la base:\n" + "\n".join(filas), [
            {"origen": "catalogo", "documento": a, "descripcion": d["descripcion"]}
            for a, d in CATALOGO_DOCUMENTOS.items()
        ]

    @tool(response_format="content_and_artifact")
    def buscar_en_web(consulta: str):
        """Busca en la web, SOLO en sitios oficiales permitidos (opengroup.org, omg.org, iso.org, ieee.org,
        sei.cmu.edu), y devuelve hasta 3 resultados con título, URL y un extracto corto, marcados [web].

        Política «primero la base»: úsala ÚNICAMENTE si ya llamaste a buscar_en_base y esta no tiene la
        respuesta, Y la pregunta trata de información pública o actual (una versión reciente, una
        norma vigente, una definición general). Nunca para temas que la base ya responde, ni para saludos.
        La consulta sale hacia un servicio externo: escribe una consulta general y corta, sin copiar
        texto de la base ni datos del usuario.

        Args:
            consulta: texto de búsqueda breve (máx. 200 caracteres).
        """
        consulta = " ".join(str(consulta).split())[:MAX_CHARS_CONSULTA_WEB]
        if not consulta:
            return "ERROR: consulta vacía.", []
        clave = os.environ.get("TAVILY_API_KEY")
        if not clave:
            return "WEB_NO_DISPONIBLE: la búsqueda web no está configurada (falta TAVILY_API_KEY).", []
        try:
            r = requests.post(
                TAVILY_URL,
                headers={"Authorization": f"Bearer {clave}"},
                json={
                    "query": consulta,
                    "max_results": WEB_MAX_RESULTADOS,
                    "include_domains": list(WEB_DOMINIOS_PERMITIDOS),  # fija en código
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_raw_content": False,
                    "include_images": False,
                },
                timeout=WEB_TIMEOUT_S,
            )
            r.raise_for_status()
            crudos = r.json().get("results", [])
        except Exception as e:  # red, cuota, clave inválida…
            return f"WEB_NO_DISPONIBLE: no se pudo consultar el buscador ({type(e).__name__}).", []

        resultados = []
        for x in crudos:
            url = str(x.get("url", ""))
            host = (urlparse(url).hostname or "").lower()
            # Defensa extra: se descarta cualquier resultado fuera de la lista permitida.
            if not any(host == d or host.endswith("." + d) for d in WEB_DOMINIOS_PERMITIDOS):
                continue
            extracto = " ".join(str(x.get("content", "")).split())
            if len(extracto) > WEB_MAX_CHARS_EXTRACTO:
                extracto = extracto[:WEB_MAX_CHARS_EXTRACTO].rstrip() + "…"
            resultados.append({"origen": "web", "titulo": " ".join(str(x.get("title", "")).split())[:150],
                               "url": url, "extracto": extracto})
            if len(resultados) == WEB_MAX_RESULTADOS:
                break
        if not resultados:
            return "SIN_RESULTADOS_WEB: ningún sitio permitido tiene resultados para esta consulta.", []
        partes = [
            f'<resultado id="{i}" origen="web" url="{r["url"]}">\n<titulo>{_escapar(r["titulo"])}</titulo>\n'
            f'{_escapar(r["extracto"])}\n</resultado>'
            for i, r in enumerate(resultados, 1)
        ]
        return "\n".join(partes), resultados

    return [buscar_en_base, generar_referencia_apa, listar_documentos_base, buscar_en_web]

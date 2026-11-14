"""
Avance 3 — Evaluación del AGENTE.

Tres partes (se pueden correr por separado):
  1. Ragas: las mismas 21 preguntas del Avance 2 (16 dentro del corpus) sobre el agente. Los retrieved_contexts son los
     fragmentos que devolvieron sus tools. Guarda resultados/agente_*.json|csv y compara con el Avance 2 (`base`).
  2. Costo: para las mismas preguntas mide llamadas al LLM, tokens y tiempo del pipeline fijo (Avance 2) y del agente.
  3. Casos de comportamiento: eval/casos_comportamiento.json (el comportamiento esperado está escrito ANTES de ejecutar).
     Registra tools usadas, llamadas al LLM y si cumplió, con comprobaciones automáticas.

Uso (desde la raíz del proyecto; requiere GROQ_API_KEY y, para Ragas, el juez; para el caso web, TAVILY_API_KEY):
    python -m eval.run_agente --casos                                   # solo los casos de comportamiento
    python -m eval.run_agente --ragas --judge-provider gemini --workers 1
    python -m eval.run_agente --costo                                   # pipeline fijo vs agente (mismas preguntas)
    python -m eval.run_agente --todo --judge-provider gemini --workers 1

Usa el MISMO modelo de Groq (GROQ_MODEL) y el MISMO juez que en el Avance 2, o la comparación no es válida.
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

os.environ.setdefault("INDEX_TAG", "base")   # reutiliza el índice del Avance 2 (misma ingesta, chunking e índice)

RES_DIR = Path(__file__).resolve().parent / "resultados"
PREGUNTAS = Path(__file__).resolve().parent / "preguntas_eval.json"
CASOS = Path(__file__).resolve().parent / "casos_comportamiento.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ragas", action="store_true", help="Parte 1: Ragas del agente")
    p.add_argument("--costo", action="store_true", help="Parte 2: costo pipeline fijo vs agente")
    p.add_argument("--casos", action="store_true", help="Parte 3: casos de comportamiento")
    p.add_argument("--todo", action="store_true", help="Las tres partes")
    p.add_argument("--tag", default="agente", help="Prefijo de los archivos de resultados")
    p.add_argument("--base-tag", default="base", help="Tag del Avance 2 con el que se compara")
    p.add_argument("--judge-provider", choices=["groq", "gemini", "ollama"], default="gemini")
    p.add_argument("--judge-model", default=os.environ.get("RAGAS_JUDGE_MODEL"))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--pausa", type=float, default=2.0, help="Segundos entre preguntas (límites de la API)")
    p.add_argument("--limit", type=int, help="Solo las primeras N preguntas (pruebas rápidas)")
    p.add_argument("--reusar-respuestas", action="store_true", help="Parte 1: conservar las respuestas ya guardadas y solo correr las que falten")
    p.add_argument("--solo-fase1", action="store_true", help="Parte 1: omitir Ragas (solo respuestas y métricas básicas)")
    a = p.parse_args()
    if a.todo:
        a.ragas = a.costo = a.casos = True
    if not (a.ragas or a.costo or a.casos):
        p.error("elige al menos una parte: --ragas, --costo, --casos o --todo")
    return a


def construir_agente():
    from src.agent import Agente
    from src.embeddings import get_embeddings
    from src.llm import get_llm
    from src.splitter import split_documents
    from src.document_loader import load_documents
    from src.vectorstore import build_vector_store, indice_existe, load_vector_store

    emb = get_embeddings()
    vs = load_vector_store(emb) if indice_existe() else build_vector_store(split_documents(load_documents()), emb)
    llm = get_llm()
    return Agente(llm, vs), vs, llm


def _tools_str(r):
    return ", ".join(f"{t['tool']}({json.dumps(t['args'], ensure_ascii=False)})" for t in r["tools_usadas"]) or "—"


# ------------------------------------------------------------------------------------------ Parte 1
def correr_agente_preguntas(a, agente, preguntas, f_resp, filas=None):
    """Corre el agente pregunta por pregunta y guarda el avance en f_resp tras CADA una (si la API
    se corta, no se pierde nada: reanuda con --reusar-respuestas)."""
    from eval.run_ragas import con_reintentos

    filas = list(filas or [])
    hechas = {f["id"] for f in filas}
    for i, q in enumerate(preguntas, 1):
        if q["id"] in hechas:
            continue
        print(f"[{i}/{len(preguntas)}] {q['id']} {q['pregunta']}")
        try:
            r = con_reintentos(lambda: agente.responder(q["pregunta"]))
        except Exception as e:
            f_resp.write_text(json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")
            sys.exit(f"\n✖ Se detuvo en {q['id']} ({type(e).__name__}). Guardé {len(filas)} respuestas en {f_resp.name}.\n"
                     "  Si fue el límite diario de Groq, espera a que se libere y reanuda con el mismo comando + --reusar-respuestas.")
        hit = None
        if q["en_corpus"]:
            esperados = {(f["documento"], p) for f in q["fuentes_esperadas"] for p in f["paginas"]}
            hit = any(rec in esperados for rec in r["documentos_recuperados"])
        filas.append({
            "id": q["id"], "pregunta": q["pregunta"], "ground_truth": q["ground_truth"],
            "en_corpus": q["en_corpus"], "categoria": q["categoria"],
            "respuesta": r["respuesta"], "contextos": r["contextos"], "recuperados": r["documentos_recuperados"],
            "sin_informacion": r["sin_informacion"], "hit_recuperacion": hit,
            "tools": r["tools_usadas"], "llamadas_llm": r["llamadas_llm"], "tokens": r["tokens"],
            "segundos": r["segundos"], "usa_web": any(t["tool"] == "buscar_en_web" for t in r["tools_usadas"]),
        })
        f_resp.write_text(json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"     tools: {_tools_str(r)} | llm: {r['llamadas_llm']} | tokens: {r['tokens']} | {r['segundos']} s")
        time.sleep(a.pausa)
    return filas


def parte_ragas(a, agente):
    from eval.run_ragas import fase2_ragas, metricas_basicas
    from src import config

    preguntas = json.loads(PREGUNTAS.read_text(encoding="utf-8"))
    if a.limit:
        preguntas = preguntas[: a.limit]
    f_resp = RES_DIR / f"{a.tag}_respuestas.json"
    previas = json.loads(f_resp.read_text(encoding="utf-8")) if (a.reusar_respuestas and f_resp.exists()) else []
    if previas:
        print(f"Reanudando: {len(previas)} respuestas ya guardadas en {f_resp.name}")
    filas = correr_agente_preguntas(a, agente, preguntas, f_resp, previas)
    f_resp.write_text(json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")

    resumen = {
        "tag": a.tag,
        "config": {"chunk_size": config.CHUNK_SIZE, "chunk_overlap": config.CHUNK_OVERLAP, "top_k": config.TOP_K,
                   "min_relevance": config.MIN_RELEVANCE, "embedding_model": config.EMBEDDING_MODEL,
                   "llm": config.GROQ_MODEL, "juez_ragas": f"{a.judge_provider}:{a.judge_model}"},
        "basicas": metricas_basicas(filas),
        "agente": {
            "preguntas_con_web": sum(f["usa_web"] for f in filas),
            "llamadas_llm_promedio": round(statistics.mean(f["llamadas_llm"] for f in filas), 2),
            "tokens_promedio": round(statistics.mean(f["tokens"] for f in filas), 1),
            "segundos_promedio": round(statistics.mean(f["segundos"] for f in filas), 2),
        },
    }
    if not a.solo_fase1:
        df = fase2_ragas(a, filas)
        df.to_csv(RES_DIR / f"{a.tag}_ragas_detalle.csv", index=False, encoding="utf-8")
        mets = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        resumen["ragas"] = {m: (round(float(df[m].mean()), 3) if m in df else None) for m in mets}
        resumen["ragas_nan"] = {m: int(df[m].isna().sum()) for m in mets if m in df}
        if any(resumen["ragas_nan"].values()):
            print("\n⚠ Hay valores NaN (límites de la API del juez). Repite con --reusar-respuestas --workers 1.")
    (RES_DIR / f"{a.tag}_resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    escribir_comparacion_ragas(a)
    print(json.dumps(resumen, ensure_ascii=False, indent=2))


def escribir_comparacion_ragas(a):
    """Tabla Avance 2 (pipeline fijo) vs Avance 3 (agente) en eval/resultados/comparacion_agente.md."""
    f_base, f_ag = RES_DIR / f"{a.base_tag}_resumen.json", RES_DIR / f"{a.tag}_resumen.json"
    if not (f_base.exists() and f_ag.exists()):
        return
    b, g = json.loads(f_base.read_text(encoding="utf-8")), json.loads(f_ag.read_text(encoding="utf-8"))
    lineas = ["| Métrica | Avance 2 (pipeline fijo) | Avance 3 (agente) | Diferencia |", "|---|---|---|---|"]
    for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        vb, vg = (b.get("ragas") or {}).get(m), (g.get("ragas") or {}).get(m)
        d = f"{vg - vb:+.3f}" if None not in (vb, vg) else "n/d"
        lineas.append(f"| {m} | {vb if vb is not None else 'n/d'} | {vg if vg is not None else 'n/d'} | {d} |")
    for m, n in (("hit_rate_recuperacion", "hit-rate de recuperación"),
                 ("abstencion_correcta_fuera_corpus", "abstención correcta (fuera del corpus)"),
                 ("abstencion_incorrecta_dentro_corpus", "abstención incorrecta (dentro)")):
        vb, vg = b["basicas"].get(m), g["basicas"].get(m)
        lineas.append(f"| {n} | {vb} | {vg} | {vg - vb:+.3f} |")
    lineas.append(f"\nPreguntas evaluadas: {g['basicas']['n_preguntas']} "
                  f"({g['basicas']['n_dentro_corpus']} dentro del corpus, {g['basicas']['n_fuera_corpus']} fuera). "
                  f"Juez Ragas: {g['config']['juez_ragas']} (Avance 2: {b['config']['juez_ragas']}).")
    (RES_DIR / "comparacion_agente.md").write_text("\n".join(lineas), encoding="utf-8")
    print("\n".join(lineas))


# ------------------------------------------------------------------------------------------ Parte 2
def parte_costo(a, agente, vs, llm):
    from eval.run_ragas import con_reintentos
    from src import config
    from src.pipeline import rag_pipeline

    preguntas = json.loads(PREGUNTAS.read_text(encoding="utf-8"))
    if a.limit:
        preguntas = preguntas[: a.limit]
    medidas = {"pipeline_fijo": [], "agente": []}
    for i, q in enumerate(preguntas, 1):
        print(f"[costo {i}/{len(preguntas)}] {q['id']}")
        r = con_reintentos(lambda: rag_pipeline(q["pregunta"], vs, llm, k=config.TOP_K,
                                                min_relevance=config.MIN_RELEVANCE, verbose=False))
        medidas["pipeline_fijo"].append({"id": q["id"], "llamadas_llm": r["llamadas_llm"], "tokens": r["tokens"],
                                         "segundos": r["segundos"]})
        time.sleep(a.pausa)
        r = con_reintentos(lambda: agente.responder(q["pregunta"]))
        medidas["agente"].append({"id": q["id"], "llamadas_llm": r["llamadas_llm"], "tokens": r["tokens"],
                                  "segundos": r["segundos"]})
        time.sleep(a.pausa)
    resumen = {k: {m: round(statistics.mean(x[m] for x in v), 2) for m in ("llamadas_llm", "tokens", "segundos")}
               for k, v in medidas.items()}
    resumen["n_preguntas"] = len(preguntas)
    resumen["detalle"] = medidas
    (RES_DIR / f"{a.tag}_costo.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    tabla = ["| Medida (promedio por pregunta) | Pipeline fijo (Avance 2) | Agente (Avance 3) |", "|---|---|---|",
             f"| Llamadas al LLM | {resumen['pipeline_fijo']['llamadas_llm']} | {resumen['agente']['llamadas_llm']} |",
             f"| Tokens | {resumen['pipeline_fijo']['tokens']} | {resumen['agente']['tokens']} |",
             f"| Tiempo de respuesta (s) | {resumen['pipeline_fijo']['segundos']} | {resumen['agente']['segundos']} |",
             f"\nPreguntas usadas: las {len(preguntas)} de eval/preguntas_eval.json (sin historial)."]
    (RES_DIR / f"{a.tag}_costo.md").write_text("\n".join(tabla), encoding="utf-8")
    print("\n".join(tabla))


# ------------------------------------------------------------------------------------------ Parte 3
def evaluar_caso(caso, r):
    """Aplica las comprobaciones de `checks` a la respuesta del agente. Devuelve (cumple, [fallos])."""
    from src.tools import WEB_DOMINIOS_PERMITIDOS

    c = caso["checks"]
    nombres = [t["tool"] for t in r["tools_usadas"]]
    resp = r["respuesta"]
    fallos = []

    proh = c.get("tools_prohibidas", [])
    usadas_prohibidas = nombres if proh == ["*"] else [n for n in nombres if n in proh]
    if usadas_prohibidas:
        fallos.append(f"usó tools prohibidas: {sorted(set(usadas_prohibidas))}")
    for t in c.get("tools_obligatorias", []):
        if t not in nombres:
            fallos.append(f"no llamó {t}")
    for t, n in c.get("min_llamadas_tool", {}).items():
        if nombres.count(t) < n:
            fallos.append(f"llamó {t} {nombres.count(t)} vez/veces (mínimo {n})")
    if c.get("orden_tools"):
        pos, ok = -1, True
        for t in c["orden_tools"]:
            sig = [i for i, n in enumerate(nombres) if n == t and i > pos]
            if not sig:
                ok = False
                break
            pos = sig[0]
        if not ok:
            fallos.append(f"orden de tools esperado {c['orden_tools']}, obtenido {nombres}")
    for t, reglas in c.get("args_deben_contener", {}).items():
        llamadas = [x["args"] for x in r["tools_usadas"] if x["tool"] == t]
        if llamadas:
            for arg, subcadenas in reglas.items():
                valor = " ".join(str(x.get(arg, "")) for x in llamadas).lower()
                faltan = [s for s in subcadenas if s.lower() not in valor]
                if faltan:
                    fallos.append(f"argumento {t}.{arg}={valor!r} sin {faltan}")
    for patron in c.get("respuesta_debe_contener", []):
        if not re.search(patron, resp, re.IGNORECASE):
            fallos.append(f"la respuesta no contiene /{patron}/")
    for patron in c.get("respuesta_no_debe_contener", []):
        if re.search(re.escape(patron), resp, re.IGNORECASE):
            fallos.append(f"la respuesta contiene «{patron}»")
    if c.get("respuesta_max_chars") and len(resp) > c["respuesta_max_chars"]:
        fallos.append(f"respuesta de {len(resp)} caracteres (máx. {c['respuesta_max_chars']})")
    if c.get("max_llamadas_llm") and r["llamadas_llm"] > c["max_llamadas_llm"]:
        fallos.append(f"{r['llamadas_llm']} llamadas al LLM (máx. {c['max_llamadas_llm']})")
    if c.get("urls_web_solo_permitidas"):
        for url in re.findall(r"\[web:\s*(https?://[^\]\s]+)", resp):
            host = (urlparse(url).hostname or "").lower()
            if not any(host == d or host.endswith("." + d) for d in WEB_DOMINIOS_PERMITIDOS):
                fallos.append(f"URL web fuera de la lista permitida: {url}")
        if not [w for w in r["fuentes_web"]]:
            fallos.append("la tool web no devolvió resultados (¿falta TAVILY_API_KEY?)")
    return (not fallos), fallos


def parte_casos(a, agente):
    from eval.run_ragas import con_reintentos

    casos = json.loads(CASOS.read_text(encoding="utf-8"))
    salida = []
    for caso in casos:
        print(f"[{caso['id']}] {caso['tipo']}: {caso['pregunta']}")
        r = con_reintentos(lambda: agente.responder(caso["pregunta"]))
        cumple, fallos = evaluar_caso(caso, r)
        salida.append({
            "id": caso["id"], "tipo": caso["tipo"], "pregunta": caso["pregunta"], "esperado": caso["esperado"],
            "obtenido": r["respuesta"], "tools": r["tools_usadas"], "llamadas_llm": r["llamadas_llm"],
            "tokens": r["tokens"], "segundos": r["segundos"], "cumple": cumple, "fallos": fallos,
            "fuentes_base": [(f["documento"], f["pagina"]) for f in r["fuentes_base"]],
            "fuentes_web": [w["url"] for w in r["fuentes_web"]],
        })
        print(f"     {'CUMPLE' if cumple else 'NO CUMPLE'} | tools: {_tools_str(r)} | llm: {r['llamadas_llm']}"
              + (f" | {fallos}" if fallos else ""))
        time.sleep(a.pausa)
    (RES_DIR / f"{a.tag}_casos.json").write_text(json.dumps(salida, ensure_ascii=False, indent=2), encoding="utf-8")

    def celda(t, n=220):
        t = " ".join(str(t).split()).replace("|", "\\|")
        return t if len(t) <= n else t[:n].rstrip() + "…"

    md = ["| # | Tipo | Pregunta | Esperado | Obtenido | Tools usadas | Llam. LLM | ¿Cumple? |", "|---|---|---|---|---|---|---|---|"]
    for s in salida:
        tools = "; ".join(f"{t['tool']}({json.dumps(t['args'], ensure_ascii=False)})" for t in s["tools"]) or "ninguna"
        md.append(f"| {s['id']} | {s['tipo']} | {celda(s['pregunta'])} | {celda(s['esperado'])} | {celda(s['obtenido'])} | "
                  f"{celda(tools)} | {s['llamadas_llm']} | {'Sí' if s['cumple'] else 'No: ' + celda('; '.join(s['fallos']))} |")
    (RES_DIR / f"{a.tag}_casos.md").write_text("\n".join(md), encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\nCasos cumplidos: {sum(s['cumple'] for s in salida)}/{len(salida)}")


def main():
    a = parse_args()
    RES_DIR.mkdir(parents=True, exist_ok=True)
    agente, vs, llm = construir_agente()
    if a.casos:
        parte_casos(a, agente)
    if a.ragas:
        parte_ragas(a, agente)
    if a.costo:
        parte_costo(a, agente, vs, llm)


if __name__ == "__main__":
    sys.exit(main())

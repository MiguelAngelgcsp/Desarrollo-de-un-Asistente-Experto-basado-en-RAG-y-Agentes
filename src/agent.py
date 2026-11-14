"""
Avance 3 — Bucle del agente (manual, con tool calling de LangChain).

pregunta + historial
   -> LLM (con las tools enlazadas con bind_tools) decide: ¿responde directo o llama una tool?
   -> si llama tools: el PROGRAMA las ejecuta (src/tools.py) y le devuelve los resultados como ToolMessage
   -> vuelve al LLM… hasta que responde sin pedir más tools, o se alcanza MAX_ITERACIONES
   -> si se alcanza el límite: una llamada final SIN tools para que responda con lo recopilado

Límites en código (no en el prompt):
    MAX_ITERACIONES          llamadas al LLM con tools por pregunta
    MAX_TOOL_CALLS_PREGUNTA  tool calls ejecutadas por pregunta (el excedente se rechaza)
    MAX_BUSQUEDAS_BASE       llamadas a buscar_en_base por pregunta; una consulta repetida se rechaza
Regla «primero la base»: buscar_en_web se rechaza si en esta pregunta aún no se llamó buscar_en_base.

Registro: cada pregunta se guarda como una línea JSON en logs/uso_tools.jsonl (tools, argumentos,
llamadas al LLM, tokens, tiempo).
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from src.config import BASE_DIR, MAX_HISTORY_MESSAGES
from src.prompts import NO_INFO, SYSTEM_PROMPT_AGENTE
from src.tools import crear_tools

MAX_ITERACIONES = 5            # llamadas al LLM con tools por pregunta
MAX_TOOL_CALLS_PREGUNTA = 6    # tool calls ejecutadas por pregunta
MAX_BUSQUEDAS_BASE = 3         # llamadas a buscar_en_base por pregunta (evita bucles de consultas casi iguales)
MAX_CHARS_ARG_LOG = 300        # recorte de argumentos largos en el registro
LOG_PATH = BASE_DIR / "logs" / "uso_tools.jsonl"

_LOG_LOCK = threading.Lock()


def registrar_uso(registro: dict, ruta=LOG_PATH) -> None:
    """Agrega una línea JSON al registro de uso de tools. Nunca debe romper la respuesta."""
    try:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK, open(ruta, "a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _recortar_args(args: dict) -> dict:
    return {k: (v[:MAX_CHARS_ARG_LOG] if isinstance(v, str) else v) for k, v in (args or {}).items()}


def _a_mensajes(historial) -> list:
    msgs = []
    for m in (historial or [])[-MAX_HISTORY_MESSAGES:]:
        cls = HumanMessage if m.get("role") == "user" else AIMessage
        msgs.append(cls(content=str(m.get("content", ""))))
    return msgs


def _tokens(msg) -> int:
    meta = getattr(msg, "usage_metadata", None) or {}
    return int(meta.get("total_tokens") or (meta.get("input_tokens", 0) + meta.get("output_tokens", 0)) or 0)


class Agente:
    def __init__(self, llm, vector_store, system_prompt: str = SYSTEM_PROMPT_AGENTE, log_path=LOG_PATH):
        self.llm = llm
        self.system_prompt = system_prompt
        self.log_path = log_path
        self.tools = crear_tools(vector_store)
        self.tools_por_nombre = {t.name: t for t in self.tools}
        self.llm_con_tools = llm.bind_tools(self.tools)

    # ------------------------------------------------------------------ utilidades
    def _invocar(self, modelo, mensajes):
        """Una llamada al LLM, con un reintento (Groq a veces devuelve tool_use_failed o 429 transitorios)."""
        try:
            return modelo.invoke(mensajes)
        except Exception:
            time.sleep(2)
            return modelo.invoke(mensajes)

    def _ejecutar_tool(self, tc: dict, estado: dict) -> tuple[ToolMessage, list]:
        nombre, args = tc["name"], tc.get("args") or {}
        t0 = time.perf_counter()
        artifact: list = []
        if nombre not in self.tools_por_nombre:
            contenido = f"ERROR: la tool «{nombre}» no existe. Tools: {', '.join(self.tools_por_nombre)}."
            tm = ToolMessage(content=contenido, tool_call_id=tc["id"], name=nombre)
        elif estado["n_tool_calls"] >= MAX_TOOL_CALLS_PREGUNTA:
            contenido = "ERROR: límite de llamadas a tools alcanzado para esta pregunta. Responde con lo que ya tienes."
            tm = ToolMessage(content=contenido, tool_call_id=tc["id"], name=nombre)
        elif nombre == "buscar_en_base" and estado["n_base"] >= MAX_BUSQUEDAS_BASE:
            contenido = (f"RECHAZADO: ya hiciste {MAX_BUSQUEDAS_BASE} búsquedas en la base para esta pregunta. "
                         "Responde con los fragmentos que ya tienes o con la respuesta de falta de información.")
            tm = ToolMessage(content=contenido, tool_call_id=tc["id"], name=nombre)
        elif nombre == "buscar_en_base" and " ".join(str(args.get("consulta", "")).lower().split()) in estado["consultas"]:
            contenido = "RECHAZADO: esa consulta ya se hizo. Usa una consulta distinta o responde con lo que ya tienes."
            tm = ToolMessage(content=contenido, tool_call_id=tc["id"], name=nombre)
        elif nombre == "buscar_en_web" and not estado["uso_base"]:
            contenido = ("RECHAZADO: política «primero la base». Llama antes a buscar_en_base; "
                         "usa la web solo si la base no tiene la respuesta.")
            tm = ToolMessage(content=contenido, tool_call_id=tc["id"], name=nombre)
        else:
            estado["n_tool_calls"] += 1
            if nombre == "buscar_en_base":
                estado["uso_base"] = True
                estado["n_base"] += 1
                estado["consultas"].add(" ".join(str(args.get("consulta", "")).lower().split()))
            try:
                tm = self.tools_por_nombre[nombre].invoke(tc)   # ToolMessage con .content y .artifact
                artifact = tm.artifact or []
            except Exception as e:  # argumentos inválidos, fallo interno…
                contenido = f"ERROR al ejecutar {nombre}: {type(e).__name__}: {e}"
                tm = ToolMessage(content=contenido, tool_call_id=tc["id"], name=nombre)
        estado["trazas"].append({
            "tool": nombre, "args": _recortar_args(args),
            "n_resultados": len(artifact),
            "resultado": "ok" if artifact else str(tm.content)[:80].replace("\n", " "),
            "segundos": round(time.perf_counter() - t0, 3),
        })
        return tm, artifact

    # ------------------------------------------------------------------ API principal
    def responder(self, pregunta: str, historial=None, registrar: bool = True) -> dict:
        t_ini = time.perf_counter()
        mensajes = [SystemMessage(content=self.system_prompt), *_a_mensajes(historial), HumanMessage(content=pregunta)]
        estado = {"n_tool_calls": 0, "uso_base": False, "n_base": 0, "consultas": set(), "trazas": []}
        fuentes_base, fuentes_web, otros = [], [], []
        llamadas_llm = tokens = 0
        limite_alcanzado = False
        respuesta_msg = None

        for _ in range(MAX_ITERACIONES):
            ai = self._invocar(self.llm_con_tools, mensajes)
            llamadas_llm += 1
            tokens += _tokens(ai)
            if not getattr(ai, "tool_calls", None):
                respuesta_msg = ai
                break
            mensajes.append(ai)
            for tc in ai.tool_calls:
                tm, artifact = self._ejecutar_tool(tc, estado)
                mensajes.append(tm)
                for a in artifact:
                    {"base": fuentes_base, "web": fuentes_web}.get(a.get("origen"), otros).append(a)

        if respuesta_msg is None:  # límite de iteraciones: respuesta final sin tools
            limite_alcanzado = True
            mensajes.append(HumanMessage(content=(
                "Ya no puedes llamar más tools. Responde ahora solo con lo que devolvieron las tools; "
                f"si no alcanza, responde exactamente: \"{NO_INFO}\"")))
            respuesta_msg = self._invocar(self.llm, mensajes)
            llamadas_llm += 1
            tokens += _tokens(respuesta_msg)

        texto = respuesta_msg.content
        if isinstance(texto, list):  # algunos modelos devuelven bloques de contenido
            texto = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in texto)
        texto = (texto or "").strip() or NO_INFO
        sin_info = texto.startswith(NO_INFO[:30])

        resultado = {
            "pregunta": pregunta,
            "respuesta": texto,
            "sin_informacion": sin_info,
            "tools_usadas": [{"tool": t["tool"], "args": t["args"]} for t in estado["trazas"]],
            "trazas": estado["trazas"],
            # Si el agente se abstiene no se muestran fuentes: no respaldan ninguna respuesta.
            "fuentes_base": [] if sin_info else fuentes_base,
            "fuentes_web": [] if sin_info else fuentes_web,
            "referencias": otros,
            # Todo lo que devolvieron las tools de recuperación (para Ragas: retrieved_contexts)
            "contextos": [f["fragmento"] for f in fuentes_base] + [w["extracto"] for w in fuentes_web],
            "documentos_recuperados": [(f["documento"], f["pagina"]) for f in fuentes_base],
            "llamadas_llm": llamadas_llm,
            "tokens": tokens,
            "segundos": round(time.perf_counter() - t_ini, 3),
            "limite_iteraciones": limite_alcanzado,
        }
        if registrar:
            registrar_uso({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "pregunta": pregunta[:500],
                "tools": estado["trazas"],
                "llamadas_llm": llamadas_llm,
                "tokens": tokens,
                "segundos": resultado["segundos"],
                "limite_iteraciones": limite_alcanzado,
                "sin_informacion": sin_info,
            }, self.log_path)
        return resultado

# Base de Conocimiento: Arquitectura Empresarial (Ingeniería de Software II)

Este documento simula el "manual/guía de asignatura" que el Tutor Académico
usará como fuente de verdad. En un Avance posterior (RAG real) este contenido
se dividirá en fragmentos (chunks), se indexará con embeddings y se recuperará
dinámicamente. En este Avance 1 se usa como contexto estático inyectado en el
prompt mediante delimitadores.

## 1. Arquitectura Empresarial (AE)

La Arquitectura Empresarial es una disciplina que alinea los procesos de
negocio, la información, las aplicaciones y la infraestructura tecnológica de
una organización con su estrategia. Se organiza típicamente en capas:

- Arquitectura de Negocio (procesos, objetivos, estructura organizacional)
- Arquitectura de Información/Datos
- Arquitectura de Aplicaciones
- Arquitectura de Infraestructura/Tecnología

## 2. TOGAF (The Open Group Architecture Framework)

TOGAF es un framework para el desarrollo de arquitecturas empresariales,
estructurado alrededor del ADM (Architecture Development Method), un ciclo
iterativo de fases:

1. Fase Preliminar
2. Visión de la Arquitectura
3. Arquitectura de Negocio
4. Arquitectura de Sistemas de Información (datos y aplicaciones)
5. Arquitectura Tecnológica
6. Oportunidades y Soluciones
7. Planificación de la Migración
8. Gobierno de la Implementación
9. Gestión de Cambios de la Arquitectura

## 3. BMM (Business Motivation Model)

El BMM describe el "por qué" de la arquitectura de negocio, relacionando:

- **Metas**: fines cualitativos de largo plazo.
- **Objetivos**: fines cuantificables y medibles que concretan una meta.
- **Estrategias**: cómo se alcanzará una meta.
- **Tácticas**: acciones concretas que implementan una estrategia.
- **Influenciadores**: factores internos o externos que afectan las
  decisiones (ej: regulación, competencia, cultura organizacional).
- **Políticas**: lineamientos generales que orientan la toma de decisiones.
- **Reglas de negocio**: derivadas de las políticas, son específicas y
  verificables (ej: "todo contrato debe firmarse digitalmente en máximo 24h").

## 4. BPMN (Business Process Model and Notation)

Notación estándar para modelar procesos de negocio mediante diagramas de
flujo, con elementos como eventos (inicio/fin), actividades, compuertas de
decisión (gateways) y flujos de secuencia/mensaje. Su objetivo es que tanto
analistas de negocio como técnicos compartan una misma representación visual
del proceso.

## 5. Business Model Canvas

Herramienta de una sola página para describir un modelo de negocio a través
de 9 bloques: segmentos de clientes, propuesta de valor, canales, relación
con clientes, fuentes de ingreso, recursos clave, actividades clave, socios
clave y estructura de costos.

## 6. C4ISR

Sigla de Command, Control, Communications, Computers, Intelligence,
Surveillance and Reconnaissance. Es un modelo de arquitectura originado en el
ámbito militar para integrar sistemas de mando, control y comunicaciones con
inteligencia y vigilancia en tiempo real. Se ha adaptado al mundo empresarial
como referencia para diseñar arquitecturas de integración de sistemas
críticos donde la información debe fluir de forma centralizada y confiable
entre distintos niveles de decisión.

**Importancia y aplicaciones**: permite coordinar sistemas heterogéneos bajo
una misma cadena de mando/información, es referencia para arquitecturas de
alta disponibilidad y monitoreo, y se usa como analogía para diseñar
arquitecturas empresariales de sectores como telecomunicaciones, energía y
logística, donde la integración y la toma de decisiones en tiempo real son
críticas.

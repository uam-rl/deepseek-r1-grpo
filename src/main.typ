// --- Importaciones ---
#import "@preview/touying:0.6.1": *
#import themes.university: *

// --- Configuración del Tema ---
#show: university-theme.with(
  aspect-ratio: sys.inputs.at("aspect-ratio", default: "16-9"),
  align: horizon,
  config-common(handout: sys.inputs.at("handout", default: "false") == "true"),
  config-info(
    title: [DeepSeek R1 y GRPO],
    subtitle: [Optimización de Política con Recompensa de Grupo],
    author: [Equipo UAM RL],
  ),
  footer-a: [Equipo UAM RL],
)

// --- Configuraciones Generales ---
#set text(lang: "es")
#set text(font: "New Computer Modern")

// --- Contenido de la Presentación ---

#title-slide()

= Introducción

== ¿Qué es DeepSeek R1?

DeepSeek R1 es un modelo de lenguaje avanzado desarrollado con técnicas de razonamiento mejorado.

Características principales:
- Capacidades de razonamiento paso a paso
- Arquitectura optimizada
- Entrenamiento con RLHF
- Rendimiento competitivo

== ¿Qué es GRPO?

Group Relative Policy Optimization (GRPO) es un algoritmo de optimización de políticas.

Conceptos clave:
- Optimización relativa de grupo
- Mejora sobre PPO
- Eficiencia de muestras
- Estabilidad en entrenamiento

= GRPO en Detalle

== Fundamentos de GRPO

Principios básicos del algoritmo:

- Comparación dentro de grupos
- Normalización relativa de recompensas
- Reducción de varianza
- Mejor convergencia

== Ventajas de GRPO

Beneficios sobre métodos tradicionales:

- Mayor estabilidad
- Menos hiperparámetros sensibles
- Mejor uso de datos
- Escalabilidad mejorada

= DeepSeek R1 y GRPO

== Integración

Cómo GRPO mejora el entrenamiento de R1:

- Optimización de razonamiento
- Alineación con preferencias humanas
- Entrenamiento más eficiente
- Mejor generalización

== Resultados

Métricas de rendimiento observadas:

- Mejora en benchmarks de razonamiento
- Reducción de alucinaciones
- Mayor consistencia
- Eficiencia computacional

= Conclusión

== Resumen

Puntos clave:

- DeepSeek R1 demuestra avances en razonamiento
- GRPO ofrece optimización robusta
- Combinación prometedora para LLMs
- Futuras direcciones de investigación

Gracias por su atención!

// --- Importaciones ---
#import "@preview/touying:0.6.1": *
#import themes.university: *

// --- Configuración del Tema ---
#show: university-theme.with(
  aspect-ratio: sys.inputs.at("aspect-ratio", default: "16-9"),
  align: horizon,
  config-common(handout: sys.inputs.at("handout", default: "false") == "true"),
  config-info(
    title: [Estado del Aprendizaje por Refuerzo],
    subtitle: [Panorama a Finales de 2025],
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

== ¿Qué es el Aprendizaje por Refuerzo?

El Aprendizaje por Refuerzo (RL) es un tipo de aprendizaje automático donde un agente aprende a tomar decisiones mediante la interacción con un entorno.

Conceptos clave:
- Agente
- Entorno
- Acciones
- Recompensas
- Política

== Aplicaciones de RL

Principales áreas de aplicación en 2025:

- Robótica y sistemas autónomos
- Juegos y simulaciones
- Optimización de recursos
- Salud y descubrimiento de fármacos
- Finanzas y trading

= Avances Recientes

== RL Basado en Modelos

Progreso reciente en aprendizaje por refuerzo basado en modelos:

- Modelos del mundo mejorados
- Mejor eficiencia en muestras
- Integración con modelos de lenguaje grandes
- Enfoques híbridos que combinan métodos con y sin modelo

== Métodos de Gradiente de Política

Desarrollos clave:

- Optimización de región de confianza
- Arquitecturas actor-crítico
- Métodos de entrenamiento paralelizados
- Mejoras en estabilidad

= Direcciones Futuras

== Desafíos Abiertos

Desafíos actuales en investigación de RL:

- Eficiencia en muestras
- Generalización entre tareas
- Seguridad y robustez
- Escalado a entornos complejos
- Interpretabilidad

== Conclusión

Resumen:

- RL continúa avanzando rápidamente
- Nuevas aplicaciones emergentes
- Desafíos significativos permanecen
- Emocionantes oportunidades por delante

Gracias por su atención!

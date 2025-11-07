// --- Importaciones ---
#import "@preview/touying:0.6.1": *
#import themes.university: *

// --- Configuración del Tema ---
#show: university-theme.with(
  aspect-ratio: sys.inputs.at("aspect-ratio", default: "16-9"),
  align: horizon,
  config-common(handout: sys.inputs.at("handout", default: "false") == "true"),
  config-info(
    title: [State of Reinforcement Learning],
    subtitle: [Late 2025 Overview],
    author: [UAM RL Team],
  ),
  footer-a: [UAM RL Team],
)

// --- Configuraciones Generales ---
#set text(lang: "en")
#set text(font: "New Computer Modern")

// --- Contenido de la Presentación ---

#title-slide()

= Introduction

== What is Reinforcement Learning?

Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by interacting with an environment.

Key concepts:
- Agent
- Environment
- Actions
- Rewards
- Policy

== RL Applications

Main application areas in 2025:

- Robotics and autonomous systems
- Game playing and simulations
- Resource optimization
- Healthcare and drug discovery
- Finance and trading

= Recent Advances

== Model-Based RL

Recent progress in model-based reinforcement learning:

- Improved world models
- Better sample efficiency
- Integration with large language models
- Hybrid approaches combining model-free and model-based methods

== Policy Gradient Methods

Key developments:

- Trust region optimization
- Actor-critic architectures
- Parallelized training methods
- Stability improvements

= Future Directions

== Open Challenges

Current challenges in RL research:

- Sample efficiency
- Generalization across tasks
- Safety and robustness
- Scaling to complex environments
- Interpretability

== Conclusion

Summary:

- RL continues to advance rapidly
- New applications emerging
- Significant challenges remain
- Exciting opportunities ahead

Thank you for your attention!

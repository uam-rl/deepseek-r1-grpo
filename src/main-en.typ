// --- Importaciones ---
#import "@preview/touying:0.6.1": *
#import themes.university: *

// --- Configuración del Tema ---
#show: university-theme.with(
  aspect-ratio: sys.inputs.at("aspect-ratio", default: "16-9"),
  align: horizon,
  config-common(handout: sys.inputs.at("handout", default: "false") == "true"),
  config-info(
    title: [DeepSeek R1 and GRPO],
    subtitle: [Group Relative Policy Optimization],
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

== What is DeepSeek R1?

DeepSeek R1 is an advanced language model developed with enhanced reasoning capabilities.

Key features:
- Step-by-step reasoning capabilities
- Optimized architecture
- RLHF training
- Competitive performance

== What is GRPO?

Group Relative Policy Optimization (GRPO) is a policy optimization algorithm.

Key concepts:
- Group relative optimization
- Improvement over PPO
- Sample efficiency
- Training stability

= GRPO in Detail

== GRPO Fundamentals

Core principles of the algorithm:

- Within-group comparison
- Relative reward normalization
- Variance reduction
- Better convergence

== GRPO Advantages

Benefits over traditional methods:

- Greater stability
- Fewer sensitive hyperparameters
- Better data utilization
- Improved scalability

= DeepSeek R1 and GRPO

== Integration

How GRPO improves R1 training:

- Reasoning optimization
- Human preference alignment
- More efficient training
- Better generalization

== Results

Observed performance metrics:

- Improvement in reasoning benchmarks
- Reduced hallucinations
- Greater consistency
- Computational efficiency

= Conclusion

== Summary

Key takeaways:

- DeepSeek R1 demonstrates advances in reasoning
- GRPO offers robust optimization
- Promising combination for LLMs
- Future research directions

Thank you for your attention!

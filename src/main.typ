// --- Importaciones ---
#import "@preview/touying:0.6.1": *
#import themes.university: *
#import "@preview/colorful-boxes:1.4.3" as cb

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


== Intentos anteriores

=== PRM, ORM
+ "Solving math word problems with process-and outcome-based feedback" (DM, GSM8K)
+ "Let's verify step by step" (OAI, MATH, más anotaciones, active learning)

=== MCTS
Usar LLMs como Value models \
Se inició con CoT, ToT

- "AlphaZero-Like Tree-Search...Decoding and Training" (GSM8K, PrOntoQA)

=== AlphaZero-Like:
- States: Partial response + prompt
- Policy: LLM completions
- Value: Un modelo *distinto*

PUCT:
$
  "Select" a &= "argmax"_a (Q(s,a), + U(s,a)) \
  U(s,a)     &= c P(s,a) (sum_b N(s,b))/(1 + N(s,a))
$

```py
def expand(node):
    """Query LLM for next actions and create child nodes."""
    actions = llm_policy(node.text)
    for action, prior in actions:
        node.P[action] = prior
        if action not in node.children:
            child_text = node.text + action  # real code: add newline, etc.
            node.children[action] = Node(child_text, parent=node)
    node.is_expanded = True

def mcts(root, num_sims):
  for _ in range(num_sims):
    note = root
    path = []
    while node.is_expanded and len(node.P) > 0:
      action = select_child(node)
      path.append((node,action))
      # select
      node = node.children[action]

    if not node.is_expanded:
      expand(node)

    v = value_model(node.text)

    backup(path, v)

  return root

def backup(path, value):
    """
    path: list of (node, action) from root to leaf
    value: scalar from value model
    """
    for node, action in path:
        node.N[action] += 1
        node.total_visits += 1
        # incremental mean
        n = node.N[action]
        node.Q[action] += (value - node.Q[action]) / n

```


== REINFORCE

Objetivo: maximize expected return of a stochastic policy $pi_alpha (a | s)$
$
  J(theta) = E_tau [R(tau)]
$

$
  p_theta (tau) = p(s_0)
  product_t pi_theta (alpha_t | s_t) p(s_(t+1) | s_t, a_t)
$

#pagebreak(weak:true)

Notemos
$gradient_theta p_theta (tau) = p_theta (tau) gradient_theta log p_theta (tau)$,
entonces

$
  J(theta) &= sum_(tau) p_theta (tau) R(tau) \
  gradient_theta J(theta)
  &= sum_tau gradient_theta p_theta (tau) R (tau) \
  &= sum_tau p_theta (tau) gradient_theta log p_theta (tau) R (tau) \
  &= EE_(tau ~ pi_theta)[gradient_theta log p_theta (tau) R (tau)]
$

#pagebreak(weak: true)
Expandamos $log p_theta (tau)$:
$

  log p_theta (tau) = log p(s_0)
  + sum_t log pi_theta (alpha_t | s_t) + log p(s_(t+1) | s_t, a_t)\
  gradient_theta log p_theta (tau) = sum_t gradient_theta log pi_theta (alpha_t | s_t)
$

#pagebreak(weak: true)

== Teorema del Gradiente de Política

El teorema fundamental:

$
  gradient_theta J(theta) = EE_tau [
    R(tau) sum_t gradient_theta log pi_theta (a_t | s_t)
  ]
$

== Causalidad y Retornos

Por causalidad, las acciones en el tiempo $t$ solo afectan las recompensas
futuras. \
Definimos el retorno desde el tiempo $t$: $G_t = sum_(k>=t) r_k$

Esto nos permite reescribir el teorema considerando solo las recompensas futuras relevantes.

#cb.outline-colorbox(
  title: "Teorema del Gradiente de Política (con retornos)",
  color: "blue",
  width: auto,
  radius: 2pt,
  centering: true,
)[
  #v(1em)
  $
    gradient_theta J(theta) = EE_tau [
      sum_t G_t gradient_theta log pi_theta (a_t | s_t)
    ]
  $
  #v(1em)
]

#pagebreak(weak: true)





== GRPO

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

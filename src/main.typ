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

Objetivo: maximize expected return of a stochastic policy $pi_theta (a | s)$
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

== REINFORCE con baseline

Tenemos
$
  EE_(s_t, a_t) [gradient_theta log pi_theta (a_t | s_t) b(s_t)]
  &= EE_(s_t)[
    EE_(a_t|s_t) [gradient_theta log pi_theta (a_t | s_t) b(s_t) | s_t]
  ]\
  &=  EE_(s_t)[
    sum_(a_t) pi_theta (a_t | s_t) gradient_theta log pi_theta (a_t | s_t)
  ] \
  &=  EE_(s_t)[
    gradient_theta sum_(a_t) pi_theta (a_t | s_t)
  ] \
  &=  EE_(s_t)[
    1
  ] \
  &= 0
$
#pagebreak(weak: true)

Lo cual nos permite usar una fórmula usualmente con menos varianza tomando
$b(dot.c)$ adecuada

$
  EE[gradient_theta log pi_theta (a_t | s_t)(G_t - b(s_t))]
  = EE[gradient_theta log pi_theta (a_t | s_t)G_t]
$

=== Ventaja
Concretamente, podemos usar:
$
  A_t = Q_pi (s_t, a_t) - V_pi (s_t)
$

== Ejemplo conceptual: CartPole

- Consigue $s_0$
En cada $t$
- Dale a la red neuronal (política) una $s_t$ para muestrear una $a_t$
- Obten $r_t, s_(t+1)$

Al finál obtendrás una trallectoria, $tau$.

Luego, para cada paso de la trallectoria, calcula $G_t$, y usa una value
function paramétrica y estima la advantage:
$
  hat(A_t) = G_t - V_phi (s_t)
$
#pagebreak(weak: true)

Entonces, nuestras funciones loss son:
$
  L_"policy" &= - 1/T sum_(t = 0)^(T-1) log pi_theta (a_t | s_t) \
  L_"value"  &= - 1/T sum_(t = 0)^(T-1) (V_phi (s_t) - G_t)^2
$

#pagebreak(weak: true)

Entonces, para optimizar:
+ Tomamos un paso de gradiente para minimizar $L_"policy"$
+ Tomamos un paso de gradiente para minimizar $L_"value"$
+ Repetir con nuevas roll outs


== Algo de terminilogía

- On-policy: Los datos que usamos para actualizar la política son generados por la
  misma política

- Episódico: en CartPole, el juego se termina/es finito, ya sea cuando el palo
  se cae o cuando llegamos a un $T$ máximo.
  
== PPO

La mayoría de los métodos de gradiente de política funcionan calculando un estimador del gradiente de política de la forma:

$
  hat(g) = hat(EE)_t [gradient_theta log pi_theta (a_t | s_t) A_t]
$

Para poder actualizar nuestra politica de manera más frecuente pero mitigando el
hecho de que las roll outs vienen de otra distribución, podemos usar importance
sampling:
$
  EE_(x ~ p) [f(x)] = EE_q [p(x)/q(x) f(x)]
$

Luego
$
  EE_(pi_theta) [hat(A_t)]
  = EE_(pi_"old") [(pi_theta (a | s))/(pi_"old" (a | s)) hat(A_t)]
$
definamos:
$
  r_(theta) = pi_theta
$

Lo que PPO propone es
$
  L_"clip" (theta) = hat(EE)_t [min(
    r_t (theta) hat(A)_t,
    "clip"( r_t (theta), 1-epsilon, 1+epsilon )hat(A)_t
  )]
$
Y en addición a clipping, un pénalti KL:
$
  L_"KL" (theta)
  = hat(EE)_t [
    (pi_theta (a_t | s_t))/(pi_"old" (a_t | s_t))A_t
    - beta "KL"[pi_"old" (dot.c | s_t), pi_theta (dot.c | s_t)]
  ]
$

Finalmente, la función de pérdida completa combina estos términos:
$
  L^("CLIP"+V F+S) (theta) = hat(EE)_t [
    L_t^"CLIP" (theta) - c_1 L_t^(V F) (theta) + c_2 S[pi_theta](s_t)
  ]
$





= GRPO

== Fundamentos de GRPO

Basicamente PPO pero:

- No hay modelo crítico
- La baseline se estima de scores grupales

#pagebreak(weak: true)

Para cada pregunta $q$ (for question), se muestrea un grupo de outputs
${o_1, ... o_G}$ de $pi_"old"$ y se optimiza $pi_theta$

$
  J_"GRPO" = EE_(q, {o_1, ..., o_G}) \ [
  1/G sum_(i=1)^G (
    min(
      (pi_theta (o_i|q))/(pi_"old" (o_i|q)) A_i,
      "clip"((pi_theta (o_i|q))/(pi_"old" (o_i|q)), 1-epsilon, 1+epsilon)A_i
    )
  ) \
  - beta "KL"(pi_theta || pi_"ref")
  ]
$


== Resumen

Puntos clave:

- DeepSeek R1 demuestra avances en razonamiento
- GRPO es PPO menos critico plus group score as baseline
- GRPO ofrece optimización robusta
- Combinación prometedora para LLMs

Gracias por su atención!

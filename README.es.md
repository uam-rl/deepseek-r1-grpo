**Idiomas:** [English](README.md) | [Español](#)

---

# DeepSeek R1 y GRPO

Diapositivas de presentación sobre DeepSeek R1 y Group Relative Policy Optimization (GRPO) del equipo UAM RL.

## Ver Diapositivas

### Diapositivas HTML (Interactivas)

#### Español
- **16:9**: https://uam-rl.github.io/deepseek-r1-grpo/
- **4:3**: https://uam-rl.github.io/deepseek-r1-grpo/4-3.html

#### Inglés
- **16:9**: https://uam-rl.github.io/deepseek-r1-grpo/en.html
- **4:3**: https://uam-rl.github.io/deepseek-r1-grpo/en-4-3.html

### Descargas PDF

#### Español
- **Diapositivas 16:9**: https://uam-rl.github.io/deepseek-r1-grpo/diapositivas.pdf
- **Diapositivas 4:3**: https://uam-rl.github.io/deepseek-r1-grpo/diapositivas-4-3.pdf
- **Handout 16:9**: https://uam-rl.github.io/deepseek-r1-grpo/handout.pdf
- **Handout 4:3**: https://uam-rl.github.io/deepseek-r1-grpo/handout-4-3.pdf

#### Inglés
- **Diapositivas 16:9**: https://uam-rl.github.io/deepseek-r1-grpo/slides-en.pdf
- **Diapositivas 4:3**: https://uam-rl.github.io/deepseek-r1-grpo/slides-en-4-3.pdf
- **Handout 16:9**: https://uam-rl.github.io/deepseek-r1-grpo/handout-en.pdf
- **Handout 4:3**: https://uam-rl.github.io/deepseek-r1-grpo/handout-en-4-3.pdf

## Compilación Local

### Requisitos Previos
- [Typst](https://github.com/typst/typst) para compilación PDF
- [Python 3.11+](https://www.python.org/) y `touying` para generación HTML

### Comandos de Compilación

```bash
# Diapositivas PDF en español
typst compile src/main.typ build/diapositivas.pdf
typst compile src/main.typ build/diapositivas-4-3.pdf --input aspect-ratio="4-3"

# Diapositivas PDF en inglés
typst compile src/main-en.typ build/slides-en.pdf
typst compile src/main-en.typ build/slides-en-4-3.pdf --input aspect-ratio="4-3"

# Handouts en español
typst compile src/main.typ build/handout.pdf --input handout="true"
typst compile src/main.typ build/handout-4-3.pdf --input handout="true" --input aspect-ratio="4-3"

# Handouts en inglés
typst compile src/main-en.typ build/handout-en.pdf --input handout="true"
typst compile src/main-en.typ build/handout-en-4-3.pdf --input handout="true" --input aspect-ratio="4-3"

# Diapositivas HTML (requiere touying)
pip install -r requirements.txt
# Español
touying compile --root=src --output="build/index.html" src/main.typ
touying compile --root=src --output="build/4-3.html" --sys-inputs '{"aspect-ratio":"4-3"}' src/main.typ
# Inglés
touying compile --root=src --output="build/en.html" src/main-en.typ
touying compile --root=src --output="build/en-4-3.html" --sys-inputs '{"aspect-ratio":"4-3"}' src/main-en.typ
```

## Despliegue

Las diapositivas se compilan y despliegan automáticamente a GitHub Pages mediante GitHub Actions cuando se hacen cambios en la rama `master`.

**El despliegue se activa con cambios en:**
- `src/**` - Archivos fuente
- `.github/workflows/**` - Configuración del workflow
- `requirements.txt` - Dependencias de Python

## Estructura del Repositorio

```
.
├── .github/
│   └── workflows/
│       └── deploy.yml       # Workflow de GitHub Actions
├── src/
│   ├── main.typ            # Fuente de presentación en español
│   └── main-en.typ         # Fuente de presentación en inglés
├── build/                  # Archivos generados (solo local)
└── requirements.txt        # Dependencias de Python para generación HTML
```

## Licencia

Esta presentación es parte del trabajo de investigación del equipo UAM RL.

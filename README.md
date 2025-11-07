# State of Reinforcement Learning - Late 2025 Overview

Presentation slides for the UAM RL Team's overview of the state of reinforcement learning in late 2025.

Available in Spanish and English.

---

# Estado del Aprendizaje por Refuerzo - Panorama a Finales de 2025

Diapositivas de presentación del equipo UAM RL sobre el estado del aprendizaje por refuerzo a finales de 2025.

Disponible en español e inglés.

---

## View Slides / Ver Diapositivas

### HTML Slides (Interactive) / Diapositivas HTML (Interactivas)

#### Spanish / Español
- **16:9**: https://uam-rl.github.io/state-of-rl-late-2025/
- **4:3**: https://uam-rl.github.io/state-of-rl-late-2025/4-3.html

#### English / Inglés
- **16:9**: https://uam-rl.github.io/state-of-rl-late-2025/en.html
- **4:3**: https://uam-rl.github.io/state-of-rl-late-2025/en-4-3.html

### PDF Downloads / Descargas PDF

#### Spanish / Español
- **Diapositivas 16:9**: https://uam-rl.github.io/state-of-rl-late-2025/diapositivas.pdf
- **Diapositivas 4:3**: https://uam-rl.github.io/state-of-rl-late-2025/diapositivas-4-3.pdf
- **Handout 16:9**: https://uam-rl.github.io/state-of-rl-late-2025/handout.pdf
- **Handout 4:3**: https://uam-rl.github.io/state-of-rl-late-2025/handout-4-3.pdf

#### English / Inglés
- **Slides 16:9**: https://uam-rl.github.io/state-of-rl-late-2025/slides-en.pdf
- **Slides 4:3**: https://uam-rl.github.io/state-of-rl-late-2025/slides-en-4-3.pdf
- **Handout 16:9**: https://uam-rl.github.io/state-of-rl-late-2025/handout-en.pdf
- **Handout 4:3**: https://uam-rl.github.io/state-of-rl-late-2025/handout-en-4-3.pdf

## Building Locally

### Prerequisites
- [Typst](https://github.com/typst/typst) for PDF compilation
- [Python 3.11+](https://www.python.org/) and `touying` for HTML generation

### Build Commands

```bash
# Spanish PDF slides
typst compile src/main.typ build/diapositivas.pdf
typst compile src/main.typ build/diapositivas-4-3.pdf --input aspect-ratio="4-3"

# English PDF slides
typst compile src/main-en.typ build/slides-en.pdf
typst compile src/main-en.typ build/slides-en-4-3.pdf --input aspect-ratio="4-3"

# Spanish handouts
typst compile src/main.typ build/handout.pdf --input handout="true"
typst compile src/main.typ build/handout-4-3.pdf --input handout="true" --input aspect-ratio="4-3"

# English handouts
typst compile src/main-en.typ build/handout-en.pdf --input handout="true"
typst compile src/main-en.typ build/handout-en-4-3.pdf --input handout="true" --input aspect-ratio="4-3"

# HTML slides (requires touying)
pip install -r requirements.txt
# Spanish
touying compile --root=src --output="build/index.html" src/main.typ
touying compile --root=src --output="build/4-3.html" --sys-inputs '{"aspect-ratio":"4-3"}' src/main.typ
# English
touying compile --root=src --output="build/en.html" src/main-en.typ
touying compile --root=src --output="build/en-4-3.html" --sys-inputs '{"aspect-ratio":"4-3"}' src/main-en.typ
```

## Deployment

The slides are automatically built and deployed to GitHub Pages via GitHub Actions when changes are pushed to the `master` branch.

**Deployment triggers on changes to:**
- `src/**` - Source files
- `.github/workflows/**` - Workflow configuration
- `requirements.txt` - Python dependencies

## Repository Structure

```
.
├── .github/
│   └── workflows/
│       └── deploy.yml       # GitHub Actions workflow
├── src/
│   ├── main.typ            # Spanish presentation source
│   └── main-en.typ         # English presentation source
├── build/                  # Generated files (local only)
└── requirements.txt        # Python dependencies for HTML generation
```

## License

This presentation is part of the UAM RL Team's research work.

# AiAndMachineLearning

Recopilatorio de ejemplos de aplicaciones relacionadas con Machine Learning e Inteligencia Artificial.

## Requisitos

- Python 3
- Entorno virtual (`venv`)
- [Google Colab](https://colab.research.google.com/) (opcional, para notebooks)
- Google Coral TPU por USB (opcional, para inferencia)

## Estructura del proyecto

```text
.
├── Agents.md          # Convenciones y flujo de trabajo para agentes/desarrolladores
├── LICENSE            # GNU GPL v3
├── README.md
├── docs/              # Documentación
├── src/               # Código fuente y notebooks
│   ├── AI/            # Proyectos de Inteligencia Artificial
│   │   └── Colab/
│   │       └── Simple_Tokenizer/
│   └── ML/            # Proyectos de Machine Learning
│       ├── ExtractionDataAnalysis/
│       ├── K-means/
│       ├── LinearRegression2var/
│       ├── NaiveBayes/
│       ├── Recomendador/
│       └── RegresionLogistica/
└── tests/             # Tests
```

## Primeros pasos

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd AiAndMachineLearning

# Crear y activar el entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias del ejemplo que vayas a ejecutar
# (ver la documentación de cada ejemplo en docs/)
```

Guía detallada: [docs/setup.md](docs/setup.md).

## Ejemplos

| Ejemplo | Ubicación | Descripción |
|---------|-----------|-------------|
| Tokenizador simple | [`src/AI/Colab/Simple_Tokenizer/`](src/AI/Colab/Simple_Tokenizer/) | Tokenización de texto con Keras `Tokenizer` |
| Extracción y análisis de datos | [`src/ML/ExtractionDataAnalysis/`](src/ML/ExtractionDataAnalysis/) | Exploración, correlación, outliers y comparación de población |
| K-Means | [`src/ML/K-means/`](src/ML/K-means/) | Clustering de perfiles de personalidad (curva del codo y centroides) |
| Regresión lineal 2 variables | [`src/ML/LinearRegression2var/`](src/ML/LinearRegression2var/) | Predicción de shares a partir de word count y engagement |
| Regresión logística | [`src/ML/RegresionLogistica/`](src/ML/RegresionLogistica/) | Clasificación de SO (Windows / Mac / Linux) y ejemplo sintético |
| Naive Bayes | [`src/ML/NaiveBayes/`](src/ML/NaiveBayes/) | Clasificador comprar vs alquilar vivienda |
| Recomendador | [`src/ML/Recomendador/`](src/ML/Recomendador/) | Recomendación de repositorios GitHub con vecinos cercanos |

Más detalle en [docs/](docs/).

## Flujo de trabajo

1. Crear rama `feature/nombre-descripcion`
2. Implementar el código
3. Hacer commit atómico
4. Crear Pull Request
5. Esperar code review

Antes de cada commit: ejecutar linters, pasar tests y actualizar la documentación. Las convenciones de código están en [`Agents.md`](Agents.md).

## Licencia

Este proyecto se distribuye bajo la [GNU General Public License v3](LICENSE).

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
│       └── K-means/
│       └── ExtractionDataAnalysis/
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

## Ejemplos

| Ejemplo | Ubicación | Descripción |
|---------|-----------|-------------|
| Tokenizador simple | [`src/AI/Colab/Simple_Tokenizer/`](src/AI/Colab/Simple_Tokenizer/) | Tokenización de texto con Keras `Tokenizer` |
| K-Means | [`src/ML/K-means/`](src/ML/K-means/) | Clustering de perfiles de personalidad (curva del codo y centroides) |
| Extracción y análisis de datos | [`src/ML/ExtractionDataAnalysis/`](src/ML/ExtractionDataAnalysis/) | Exploración, correlación, outliers y comparación de población |

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

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
├── AGENTS.md          # Convenciones y flujo de trabajo para agentes/desarrolladores
├── LICENSE            # GNU GPL v3
├── README.md
├── docs/              # Documentación
├── src/               # Código fuente y notebooks
│   ├── AI/            # Proyectos de Inteligencia Artificial
│   │   ├── Colab/
│   │   │   ├── ConvNeuralTensorFlow_TransferLearning/
│   │   │   └── Simple_Tokenizer/
│   │   └── StandAlone/
│   │       ├── ConvNeuralTensorFlow/
│   │       └── FirstNeuralTensorFlow/
│   └── ML/            # Proyectos de Machine Learning
│       ├── ExtractionDataAnalysis/
│       ├── K-means/
│       ├── LinearRegression2var/
│       ├── NaiveBayes/
│       ├── Recomendador/
│       └── RegresionLogistica/
└── tests/             # Tests automatizados (pytest)
    ├── test_conv_neural_tensorflow.py
    ├── test_conv_neural_tensorflow_transfer_learning.py
    ├── test_extraction_data_analysis.py
    ├── test_first_neural_tensorflow.py
    ├── test_k_means.py
    ├── test_linear_regression_2var.py
    ├── test_naive_bayes.py
    ├── test_recomendador.py
    └── test_regresion_logistica.py
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
| [Tokenizador simple](docs/simple-tokenizer.md) | [`src/AI/Colab/Simple_Tokenizer/`](src/AI/Colab/Simple_Tokenizer/) | Tokenización de texto con Keras `Tokenizer` |
| [Transfer learning InceptionV3](docs/conv-neural-tensorflow-transfer-learning.md) | [`src/AI/Colab/ConvNeuralTensorFlow_TransferLearning/`](src/AI/Colab/ConvNeuralTensorFlow_TransferLearning/) | InceptionV3 congelado + cabeza densa (cats vs dogs) |
| [Primera red TensorFlow](docs/first-neural-tensorflow.md) | [`src/AI/StandAlone/FirstNeuralTensorFlow/`](src/AI/StandAlone/FirstNeuralTensorFlow/) | Regresión `y=2x-1` y clasificador Fashion-MNIST |
| [Red convolucional TensorFlow](docs/conv-neural-tensorflow.md) | [`src/AI/StandAlone/ConvNeuralTensorFlow/`](src/AI/StandAlone/ConvNeuralTensorFlow/) | CNN Fashion-MNIST con Conv2D y MaxPooling |
| [Extracción y análisis de datos](docs/extraction-data-analysis.md) | [`src/ML/ExtractionDataAnalysis/`](src/ML/ExtractionDataAnalysis/) | Exploración, correlación, outliers y comparación de población |
| [K-Means](docs/k-means.md) | [`src/ML/K-means/`](src/ML/K-means/) | Clustering de perfiles de personalidad (curva del codo y centroides) |
| [Regresión lineal 2 variables](docs/linear-regression-2var.md) | [`src/ML/LinearRegression2var/`](src/ML/LinearRegression2var/) | Predicción de shares a partir de word count y engagement |
| [Regresión logística](docs/regresion-logistica.md) | [`src/ML/RegresionLogistica/`](src/ML/RegresionLogistica/) | Clasificación de SO (Windows / Mac / Linux) y ejemplo sintético |
| [Naive Bayes](docs/naive-bayes.md) | [`src/ML/NaiveBayes/`](src/ML/NaiveBayes/) | Clasificador comprar vs alquilar vivienda |
| [Recomendador](docs/recomendador.md) | [`src/ML/Recomendador/`](src/ML/Recomendador/) | Recomendación de repositorios GitHub con vecinos cercanos |

Índice completo en [docs/README.md](docs/README.md).

## Flujo de trabajo

1. Actualiza el repositorio
2. Crear rama `feature/nombre-descripcion` (o `fix/...` para correcciones)
3. Implementar el código
4. Hacer commit atómico
5. Crear Pull Request
6. Esperar code review

Antes de cada commit: ejecutar linters, pasar tests (`MPLBACKEND=Agg pytest tests/ -v`) y actualizar la documentación. Las convenciones de código están en [`AGENTS.md`](AGENTS.md).

## Licencia

Este proyecto se distribuye bajo la [GNU General Public License v3](LICENSE).

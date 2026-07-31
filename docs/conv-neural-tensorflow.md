# Red convolucional con TensorFlow

Ejemplo **standalone** de una CNN (convolución + pooling) que clasifica
prendas de Fashion-MNIST. Las capas convolucionales detectan patrones
locales independientemente de dónde esté el objeto en la imagen.

## Ubicación

```text
src/AI/StandAlone/ConvNeuralTensorFlow/
├── requirements.txt
└── src/
    └── convNeuralTensorFlow.py
```

| Recurso | Propósito |
|---------|-----------|
| `src/convNeuralTensorFlow.py` | CNN Fashion-MNIST con early-stop por accuracy, save/load |
| `requirements.txt` | `tensorflow`, `numpy` |
| `tests/test_conv_neural_tensorflow.py` | Tests pytest (datos sintéticos, pocas épocas) |

## Objetivo

1. Construir una red secuencial con `Conv2D`, `MaxPooling2D` y capas densas
2. Entrenar con un callback que detiene el entrenamiento al alcanzar un umbral
3. Normalizar píxeles a `[0, 1]` y añadir el canal de profundidad `(28, 28, 1)`
4. Evaluar, predecir y persistir el modelo (`save` / `load`)

## Dependencias

- Python 3
- TensorFlow / Keras
- NumPy

```bash
source .venv/bin/activate
pip install -r src/AI/StandAlone/ConvNeuralTensorFlow/requirements.txt
```

La primera ejecución del script descarga Fashion-MNIST vía Keras (requiere red).

## Cómo ejecutarlo

Desde la raíz del repositorio, con el venv activado:

```bash
python src/AI/StandAlone/ConvNeuralTensorFlow/src/convNeuralTensorFlow.py
```

### Entrenamiento corto (API)

```python
import sys
import numpy as np

sys.path.insert(0, "src/AI/StandAlone/ConvNeuralTensorFlow/src")
from convNeuralTensorFlow import Modelo

imgs = np.random.randint(0, 256, size=(32, 28, 28), dtype=np.uint8)
labels = np.random.randint(0, 10, size=(32,), dtype=np.int32)

modelo = Modelo()
modelo.compile()
modelo.train(imgs, labels, epochs=2)
print(modelo.predict(imgs[0:1]).shape)  # (1, 10)
```

## Constantes relevantes

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `DEFAULT_EPOCHS` | `50` | Épocas máximas en la demo CLI |
| `ACCURACY_STOP_THRESHOLD` | `0.9` | Accuracy sobre el que el callback para |
| `NUM_DEMO_PREDICTIONS` | `3` | Predicciones impresas al final de `__main__` |

## Arquitectura

```text
Input(28, 28, 1)
  → Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)
  → Conv2D(64, 3×3, ReLU) → MaxPooling2D(2×2)
  → Flatten → Dense(128, ReLU) → Dense(10, softmax)
```

- Optimizador: Adam
- Loss: `sparse_categorical_crossentropy`
- Métrica: accuracy
- Preprocesado en `prepareData`: `/ 255.0` y `expand_dims` si falta el canal

## Tests

```bash
source .venv/bin/activate
pip install -r src/AI/StandAlone/ConvNeuralTensorFlow/requirements.txt
MPLBACKEND=Agg pytest tests/test_conv_neural_tensorflow.py -v
```

Los tests usan datos sintéticos y pocas épocas (no descargan Fashion-MNIST).
Si TensorFlow no está instalado, se saltan con `skipif`.

| Test | Qué comprueba |
|------|----------------|
| `testPrepareDataNormalizesAndAddsChannel` | Normalización y shape `(N, 28, 28, 1)` |
| `testPrepareDataKeepsExistingChannel` | No duplica el canal si ya existe |
| `testBuildCompileAndPredictShape` | Shape `(1, 10)` y probs ≈ 1 |
| `testPredictUsesNormalizedInput` | `predict` normaliza antes de inferir |
| `testSaveAndLoad` | Persistencia `save` / `load` |
| `testEvaluateReturnsLossAndAccuracy` | `evaluate` devuelve loss y accuracy |

# Primera red neuronal con TensorFlow

Ejemplo **standalone** (scripts locales, sin notebook Colab) con dos modelos
Keras: una red mínima que aprende `y = 2x - 1` y un MLP que clasifica prendas
de Fashion-MNIST.

## Ubicación

```text
src/AI/StandAlone/FirstNeuralTensorFlow/
├── requirements.txt
└── src/
    ├── intro.py
    └── cloth_detector.py
```

| Recurso | Propósito |
|---------|-----------|
| `src/intro.py` | Regresión lineal con `Dense(1)`, early-stop por MAE, save/load |
| `src/cloth_detector.py` | Clasificador Fashion-MNIST (Flatten → Dense 128 → Dense 10) |
| `requirements.txt` | `tensorflow`, `numpy` |
| `tests/test_first_neural_tensorflow.py` | Tests pytest (datos sintéticos, pocas épocas) |

## Objetivo

1. Construir y compilar modelos Keras secuenciales
2. Entrenar con callbacks que detienen el entrenamiento al alcanzar un umbral
3. Guardar / cargar el modelo de `intro.py` (`model.keras`)
4. Evaluar y predecir normalizando píxeles a `[0, 1]` en el detector

## Dependencias

- Python 3
- TensorFlow / Keras
- NumPy

```bash
source .venv/bin/activate
pip install -r src/AI/StandAlone/FirstNeuralTensorFlow/requirements.txt
```

La primera ejecución de `cloth_detector.py` descarga Fashion-MNIST vía Keras
(requiere red).

## Cómo ejecutarlo

Desde la raíz del repositorio, con el venv activado:

```bash
# Regresión y = 2x - 1 (puede tardar hasta alcanzar MAE < 1.0)
python src/AI/StandAlone/FirstNeuralTensorFlow/src/intro.py

# Clasificador Fashion-MNIST (descarga el dataset la primera vez)
python src/AI/StandAlone/FirstNeuralTensorFlow/src/cloth_detector.py
```

### Entrenamiento corto (API)

```python
import sys
import numpy as np

sys.path.insert(0, "src/AI/StandAlone/FirstNeuralTensorFlow/src")
from intro import Modelo, createTrainDataset

model = Modelo()
model.new()
model.compila()
xs, ys = createTrainDataset()
model.entrena(xs, ys, epochs=100)
print(model.prediccion(np.array([10.0], dtype=float)))
```

```python
import sys
import numpy as np

sys.path.insert(0, "src/AI/StandAlone/FirstNeuralTensorFlow/src")
from cloth_detector import Modelo

imgs = np.random.randint(0, 256, size=(32, 28, 28), dtype=np.uint8)
labels = np.random.randint(0, 10, size=(32,), dtype=np.int32)

modelo = Modelo()
modelo.compile()
modelo.train(imgs, labels, epochs=2)
print(modelo.predict(imgs[0:1]).shape)  # (1, 10)
```

## Constantes relevantes

### `intro.py`

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `DEFAULT_EPOCHS` | `50000` | Épocas máximas en la demo CLI |
| `DEFAULT_MODEL_PATH` | `model.keras` | Ruta por defecto de save/load |
| `MAE_STOP_THRESHOLD` | `1.0` | MAE bajo el cual el callback para el entrenamiento |

### `cloth_detector.py`

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `DEFAULT_EPOCHS` | `50` | Épocas máximas en la demo CLI |
| `ACCURACY_STOP_THRESHOLD` | `0.95` | Accuracy sobre el que el callback para |
| `NUM_DEMO_PREDICTIONS` | `3` | Predicciones impresas al final de `__main__` |

## Arquitectura

### `intro.py` — regresión `y = 2x - 1`

- Entrada escalar → `Dense(1)` (peso y sesgo aprenden ≈ `2` y `-1`)
- Optimizador por defecto: Adam; loss MSE; métrica MAE
- Dataset sintético: `createTrainDataset(maxTrain=100)` con `x ∈ [-1000, 1000]`

### `cloth_detector.py` — Fashion-MNIST

- `Input(28, 28)` → Flatten → Dense(128, ReLU) → Dense(10, softmax)
- Loss: `sparse_categorical_crossentropy`; métrica: accuracy
- Normalización `/ 255.0` en `train`, `evaluate` y `predict`

## Tests

```bash
source .venv/bin/activate
pip install -r src/AI/StandAlone/FirstNeuralTensorFlow/requirements.txt
MPLBACKEND=Agg pytest tests/test_first_neural_tensorflow.py -v
```

Los tests usan datos sintéticos y pocas épocas (no descargan Fashion-MNIST).
Si TensorFlow no está instalado, se saltan con `skipif`.

| Test | Qué comprueba |
|------|----------------|
| `testCreateTrainDatasetShapes` | Formas y relación `y = 2x - 1` |
| `testIntroTrainAndPredictApproximatesFormula` | Predicción cercana a la fórmula tras entrenar |
| `testIntroSaveAndLoad` | Persistencia `save` / `load` |
| `testClothDetectorBuildCompileAndPredictShape` | Shape `(1, 10)` y probs ≈ 1 |
| `testClothDetectorPredictUsesNormalizedInput` | `predict` normaliza antes de inferir |

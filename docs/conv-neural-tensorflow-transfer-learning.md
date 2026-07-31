# Transfer learning con InceptionV3 (cats vs dogs)

Ejemplo en **Google Colab** (también usable en local) de transfer learning:
InceptionV3 preentrenado en ImageNet, capas congeladas, y una cabeza densa
para clasificar gatos frente a perros.

## Ubicación

```text
src/AI/Colab/ConvNeuralTensorFlow_TransferLearning/
├── requirements.txt
└── src/
    ├── ConvNeuralTensorFlow_TransferLearning.ipynb
    └── convNeuralTensorFlowTransferLearning.py
```

| Recurso | Propósito |
|---------|-----------|
| `src/ConvNeuralTensorFlow_TransferLearning.ipynb` | Notebook didáctico (descarga Cats vs Dogs, entrena) |
| `src/convNeuralTensorFlowTransferLearning.py` | API reutilizable para tests y uso local |
| `requirements.txt` | `tensorflow`, `numpy`, `pillow` |
| `tests/test_conv_neural_tensorflow_transfer_learning.py` | Tests pytest (datos sintéticos, `weights=None`) |

## Objetivo

1. Descargar y repartir el dataset Microsoft Cats vs Dogs (train/test)
2. Cargar InceptionV3 sin la cabeza de ImageNet (`include_top=False`)
3. Congelar el backbone y tomar la salida de la capa `mixed7`
4. Añadir `Flatten` → `Dense(1024, ReLU)` → `Dense(1, sigmoid)`
5. Entrenar solo la cabeza con data augmentation y early-stop por accuracy

## Dependencias

- Python 3
- TensorFlow / Keras
- NumPy
- Pillow (lectura de imágenes en los generators)

```bash
source .venv/bin/activate
pip install -r src/AI/Colab/ConvNeuralTensorFlow_TransferLearning/requirements.txt
```

La primera ejecución del notebook descarga el ZIP de Cats vs Dogs y, si usas
`weights='imagenet'`, los pesos de InceptionV3 (requiere red).

## Cómo ejecutarlo

### En Google Colab

1. Abre el notebook en Colab
2. Ejecuta las celdas en orden (usa `/tmp` como `DOWNLOAD_DIR`)

### En local (notebook)

```bash
source .venv/bin/activate
pip install -r src/AI/Colab/ConvNeuralTensorFlow_TransferLearning/requirements.txt jupyter
jupyter notebook src/AI/Colab/ConvNeuralTensorFlow_TransferLearning/src/ConvNeuralTensorFlow_TransferLearning.ipynb
```

### API local (módulo)

```python
import sys
import numpy as np

sys.path.insert(
  0, "src/AI/Colab/ConvNeuralTensorFlow_TransferLearning/src"
)
from convNeuralTensorFlowTransferLearning import Modelo

modelo = Modelo()
modelo.define(weights=None)  # o 'imagenet' para pesos reales
modelo.compile()
sample = np.random.rand(1, 150, 150, 3).astype("float32")
print(modelo.predict(sample).shape)  # (1, 1)
```

## Constantes relevantes

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `TRAINING_SIZE` | `0.9` | Proporción train / test al repartir ficheros |
| `EPOCHS` / `DEFAULT_EPOCHS` | `20` | Épocas máximas |
| `STOP_ACCURACY` | `0.9` | Accuracy sobre el que el callback para |
| `INPUT_SHAPE` / `TARGET_SIZE` | `(150, 150, 3)` / `(150, 150)` | Entrada del modelo y del generator |
| `DENSE_UNITS` | `1024` | Neuronas de la capa densa intermedia |
| `LEARNING_RATE` | `0.0001` | Learning rate de RMSprop |
| `LAST_LAYER_NAME` | `mixed7` | Capa de InceptionV3 usada como feature extractor |

## Arquitectura

```text
InceptionV3 (ImageNet, trainable=False)
  → salida de mixed7
  → Flatten → Dense(1024, ReLU) → Dense(1, sigmoid)
```

- Optimizador: RMSprop (`learning_rate=0.0001`)
- Loss: `binary_crossentropy`
- Métrica: accuracy
- Train: `ImageDataGenerator` con rescale + augmentation
- Test: solo `rescale=1/255`

## Tests

```bash
source .venv/bin/activate
pip install -r src/AI/Colab/ConvNeuralTensorFlow_TransferLearning/requirements.txt
MPLBACKEND=Agg pytest tests/test_conv_neural_tensorflow_transfer_learning.py -v
```

Los tests usan JPEGs sintéticos y `weights=None` (no descargan Cats vs Dogs
ni pesos de ImageNet). Si TensorFlow no está instalado, se saltan con `skipif`.

| Test | Qué comprueba |
|------|----------------|
| `testSplitDataIgnoresEmptyAndRespectsRatio` | Split 80/20 e ignora ficheros vacíos |
| `testStopperStopsWhenAccuracyExceeded` | El callback para al superar el umbral |
| `testStopperIgnoresMissingMetric` | No para si falta la métrica en `logs` |
| `testBuildCompileAndPredictShape` | Shape `(N, 1)` y probs en `[0, 1]` |
| `testTrainWithTinyGenerators` | Un epoch con generators locales |
| `testSaveAndLoad` | Persistencia `save` / `load` |

## Notas

- El notebook original descargaba pesos a mano; la vía actual usa
  `weights='imagenet'` en `InceptionV3`.
- El dataset Cats vs Dogs incluye algunos JPG de tamaño 0; `splitData` los omite.
- Entrenar con el dataset completo es costoso en CPU; para pruebas rápidas usa
  el módulo con `weights=None` y pocas épocas, como hacen los tests.

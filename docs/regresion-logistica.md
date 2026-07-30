# Regresión logística

Ejemplos de **clasificación** con regresión logística: uno sobre usuarios web
(sistema operativo) y otro con datos sintéticos y frontera de decisión.

## Ubicación

```text
src/ML/RegresionLogistica/
├── data/
│   └── usuarios_win_mac_lin.csv
├── regresionLogistica.py
├── regresionLogistica_v2.py
└── requirements.txt
```

| Recurso | Propósito |
|---------|-----------|
| `regresionLogistica.py` | Clasificación multiclase Windows / Mac / Linux |
| `regresionLogistica_v2.py` | Clasificación binaria con datos sintéticos y visualización |
| `data/usuarios_win_mac_lin.csv` | Dataset de comportamiento web etiquetado por SO |

## Objetivo

### `regresionLogistica.py`

1. Cargar y explorar el CSV (`duracion`, `paginas`, `acciones`, `valor`, `clase`)
2. Entrenar `LogisticRegression` para predecir el SO (`0` Windows, `1` Mac, `2` Linux)
3. Validar con hold-out (20 %) y validación cruzada K-Fold (10 folds)
4. Mostrar accuracy, matriz de confusión y classification report

Funciones reutilizables: `loadUsuariosData`, `extractFeaturesAndLabels`,
`fitLogisticModel`, `crossValidateAccuracy`, `evaluateHoldOut`, `runPipeline`.

### `regresionLogistica_v2.py`

1. Generar datos sintéticos 2D con etiqueta binaria (`feature1 + feature2 > 10`)
2. Entrenar y evaluar el modelo
3. Dibujar la frontera de decisión

Funciones reutilizables: `generateSyntheticData`, `fitBinaryLogisticModel`,
`evaluateBinaryModel`, `plotDecisionBoundary`, `runSyntheticPipeline`.

## Dependencias

- Python 3
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

```bash
source .venv/bin/activate
pip install -r src/ML/RegresionLogistica/requirements.txt
```

## Cómo ejecutarlo

Desde la raíz del repositorio, con el venv activado:

```bash
# Pipeline multiclase (abre ventanas de plot)
python src/ML/RegresionLogistica/regresionLogistica.py

# Pipeline binario sintético
python src/ML/RegresionLogistica/regresionLogistica_v2.py
```

Sin display gráfico:

```bash
MPLBACKEND=Agg python -c "
import sys
sys.path.insert(0, 'src/ML/RegresionLogistica')
from regresionLogistica import runPipeline
runPipeline(showPlots=False)
"

MPLBACKEND=Agg python -c "
import sys
sys.path.insert(0, 'src/ML/RegresionLogistica')
from regresionLogistica_v2 import runSyntheticPipeline
runSyntheticPipeline(showPlots=False)
"
```

## Dataset (`usuarios_win_mac_lin.csv`)

| Columna | Significado |
|---------|-------------|
| `duracion` | Duración de la sesión |
| `paginas` | Páginas visitadas |
| `acciones` | Acciones realizadas |
| `valor` | Valor asociado a la sesión |
| `clase` | SO: `0` Windows, `1` Mac, `2` Linux |

Nota: algunos valores de `duracion` usan punto como separador de miles
(p. ej. `1.105` → 1105 segundos); pandas los interpreta como float.

## Tests

```bash
source .venv/bin/activate
pip install -r src/ML/RegresionLogistica/requirements.txt pytest
MPLBACKEND=Agg pytest tests/test_regresion_logistica.py -v
```

Los tests no requieren display ni red; cargan el CSV local y validan la lógica
de carga, extracción de features, entrenamiento, CV y pipeline sintético.

## Salida esperada

- Resumen estadístico y conteo por clase
- Histogramas / pairplot (script v1, si `showPlots=True`)
- Score del modelo, CV accuracy, confusion matrix y classification report
- Frontera de decisión 2D (script v2, si `showPlots=True`)

## Notas

- `random_state` fijo (`7` en v1, `42` en v2) para reproducibilidad.
- Ejemplo didáctico; no persiste el modelo.
- En entornos sin display usa `MPLBACKEND=Agg`.

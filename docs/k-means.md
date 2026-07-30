# Clustering K-Means

Ejemplo didáctico de clustering no supervisado con **K-Means** sobre perfiles de
personalidad de famosos (Openness, Extraversion, Agreeableness).

## Ubicación

```text
src/ML/K-means/
├── data/
│   └── analisis.csv
├── kMeans.py
├── k-means.ipynb
└── requirements.txt
```

| Recurso | Propósito |
|---------|-----------|
| `kMeans.py` | Script ejecutable con funciones reutilizables (carga, elbow, fit, plots) |
| `k-means.ipynb` | Notebook exploratorio equivalente (Colab / Jupyter) |
| `data/analisis.csv` | Dataset de 140 usuarios con traits y categoría profesional |

## Objetivo

1. Cargar y explorar el CSV de personalidades
2. Visualizar distribuciones y relaciones entre `op`, `ex` y `ag`
3. Estimar el número óptimo de clusters con la curva del codo (Elbow)
4. Entrenar K-Means (por defecto 5 clusters) y visualizar centroides

## Dependencias

- Python 3
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

```bash
source .venv/bin/activate
pip install -r src/ML/K-means/requirements.txt
```

## Cómo ejecutarlo

Desde la raíz del repositorio, con el venv activado:

```bash
# Pipeline completo (abre ventanas de plot)
python src/ML/K-means/kMeans.py

# Sin display gráfico
MPLBACKEND=Agg python -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src/ML/K-means')
from kMeans import runPipeline
runPipeline(showPlots=False)
"
```

También puedes abrir `k-means.ipynb` en Jupyter o [Google Colab](https://colab.research.google.com/).

## Constantes relevantes

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `CSV_FILE` | `data/analisis.csv` | Ruta al dataset |
| `FEATURE_COLUMNS` | `op`, `ex`, `ag` | Features usadas en el clustering |
| `CATEGORY_COLUMN` | `categoria` | Columna de categoría profesional (solo exploración) |
| `FIRST_N_ROWS` | `20` | Filas mostradas en la exploración inicial |
| `MAX_CLUSTERS_ELBOW` | `20` | Máximo de clusters evaluados en la curva del codo |
| `DEFAULT_N_CLUSTERS` | `5` | Clusters elegidos tras inspeccionar el elbow |
| `RANDOM_STATE` | `42` | Semilla para reproducibilidad del K-Means |

## Funciones principales (`kMeans.py`)

| Función | Descripción |
|---------|-------------|
| `loadAnalisisData()` | Carga el CSV de personalidades |
| `summarizeByCategory()` | Cuenta filas por `categoria` |
| `extractFeatures()` | Devuelve la matriz `X` con `op`, `ex`, `ag` |
| `computeElbowScores()` | Calcula scores K-Means para k = 1 … max−1 |
| `fitKMeans()` | Entrena el modelo con `nClusters` |
| `predictClusters()` | Predice la etiqueta de cluster |
| `plotElbowCurve()` | Gráfica de la curva del codo |
| `plotFeatureHistograms()` | Histogramas de columnas numéricas |
| `plotPairplot()` | Pairplot coloreado por categoría |
| `plotFeatures3DByCategory()` | Scatter 3D por categoría original |
| `plotClusters3D()` | Scatter 3D de clusters y centroides |
| `plotClusterPair()` | Scatter 2D de un par de features |
| `runPipeline()` | Orquesta carga, exploración, elbow y clustering |

## Dataset

| Columna | Significado |
|---------|-------------|
| `usuario` | Identificador / handle |
| `op` | Openness |
| `co` | Conscientiousness |
| `ex` | Extraversion |
| `ag` | Agreeableness |
| `ne` | Neuroticism |
| `wordcount` | Conteo de palabras asociado |
| `categoria` | Categoría profesional (1-9) |

Categorías: Actor/Actriz, Cantante, Modelo, Tv/series, Radio, Tecnología,
Deportes, Política, Escritor.

El clustering usa las features `op`, `ex` y `ag`.

## Tests

```bash
source .venv/bin/activate
pip install -r src/ML/K-means/requirements.txt pytest
MPLBACKEND=Agg pytest tests/test_k_means.py -v
```

Los tests no requieren display ni red; cargan `analisis.csv` localmente y
ejercitan la lógica principal con `MPLBACKEND=Agg`.

## Salida esperada

- Resumen estadístico y conteo por `categoria` en consola
- Histogramas, pairplot y scatter 3D exploratorio
- Curva del codo para elegir `k`
- Centroides y etiquetas de cluster; gráficos 3D/2D con centroides marcados

Con `DEFAULT_N_CLUSTERS=5` y `RANDOM_STATE=42`, el pipeline asigna 5 grupos
distintos sobre los 140 perfiles del CSV.

## Notas

- Por inspección de la Elbow Curve el ejemplo usa **5 clusters** (`DEFAULT_N_CLUSTERS`).
- `random_state` fijo en el entrenamiento para reproducibilidad.
- Ejemplo didáctico; no persiste el modelo ni resultados.
- En entornos sin display usa `MPLBACKEND=Agg` o `runPipeline(showPlots=False)`.

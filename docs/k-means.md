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
| `data/analisis.csv` | Dataset de usuarios con traits y categoría profesional |

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

Los tests no requieren display ni red; cargan `analisis.csv` localmente.

## Salida esperada

- Resumen estadístico y conteo por `categoria` en consola
- Histogramas, pairplot y scatter 3D exploratorio
- Curva del codo para elegir `k`
- Centroides y etiquetas de cluster; gráficos 3D/2D con centroides marcados

## Notas

- Por inspección de la Elbow Curve el ejemplo usa **5 clusters** (`DEFAULT_N_CLUSTERS`).
- `random_state` fijo en el entrenamiento para reproducibilidad.
- Ejemplo didáctico; no persiste el modelo ni resultados.
- En entornos sin display usa `MPLBACKEND=Agg`.

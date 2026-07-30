# Clustering K-Means

Ejemplo didactico de clustering no supervisado con **K-Means** sobre perfiles de
personalidad de famosos (Openness, Extraversion, Agreeableness).

## Ubicacion

```text
src/ML/K-means/
├── data/
│   └── analisis.csv
├── kMeans.py
├── k-means.ipynb
└── requirements.txt
```

| Recurso | Proposito |
|---------|-----------|
| `kMeans.py` | Script ejecutable con funciones reutilizables (carga, elbow, fit, plots) |
| `k-means.ipynb` | Notebook exploratorio equivalente (Colab / Jupyter) |
| `data/analisis.csv` | Dataset de usuarios con traits y categoria profesional |

## Objetivo

1. Cargar y explorar el CSV de personalidades
2. Visualizar distribuciones y relaciones entre `op`, `ex` y `ag`
3. Estimar el numero optimo de clusters con la curva del codo (Elbow)
4. Entrenar K-Means (por defecto 5 clusters) y visualizar centroides

## Dependencias

- Python 3
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

```bash
source .venv/bin/activate
pip install -r src/ML/K-means/requirements.txt
```

## Como ejecutarlo

Desde la raiz del repositorio, con el venv activado:

```bash
# Pipeline completo (abre ventanas de plot)
python src/ML/K-means/kMeans.py

# Sin display grafico
MPLBACKEND=Agg python -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src/ML/K-means')
from kMeans import runPipeline
runPipeline(showPlots=False)
"
```

Tambien puedes abrir `k-means.ipynb` en Jupyter o [Google Colab](https://colab.research.google.com/).

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
| `categoria` | Categoria profesional (1-9) |

Categorias: Actor/Actriz, Cantante, Modelo, Tv/series, Radio, Tecnologia,
Deportes, Politica, Escritor.

El clustering usa las features `op`, `ex` y `ag`.

## Tests

```bash
source .venv/bin/activate
pip install -r src/ML/K-means/requirements.txt pytest
MPLBACKEND=Agg pytest tests/test_k_means.py -v
```

Los tests no requieren display ni red; cargan `analisis.csv` localmente.

## Salida esperada

- Resumen estadistico y conteo por `categoria` en consola
- Histogramas, pairplot y scatter 3D exploratorio
- Curva del codo para elegir `k`
- Centroides y etiquetas de cluster; graficos 3D/2D con centroides marcados

## Notas

- Por inspeccion de la Elbow Curve el ejemplo usa **5 clusters** (`DEFAULT_N_CLUSTERS`).
- `random_state` fijo en el entrenamiento para reproducibilidad.
- Ejemplo didactico; no persiste el modelo ni resultados.
- En entornos sin display usa `MPLBACKEND=Agg`.

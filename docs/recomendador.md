# Recomendador de repositorios GitHub

Ejemplo didáctico de **filtrado colaborativo** basado en vecinos cercanos
(`NearestNeighbors`) para recomendar repositorios a partir de ratings de usuarios.

## Ubicación

```text
src/ML/Recomendador/
├── data/
│   ├── ratings.csv
│   ├── repos.csv
│   └── users.csv
├── recomendador.py
├── Recomendador.ipynb
└── requirements.txt
```

| Recurso | Propósito |
|---------|-----------|
| `recomendador.py` | Script con funciones reutilizables (carga, matriz, k-NN, MSE) |
| `Recomendador.ipynb` | Notebook exploratorio equivalente (Colab / Jupyter) |
| `data/ratings.csv` | Valoraciones usuario–repositorio |
| `data/repos.csv` | Catálogo de repositorios |
| `data/users.csv` | Catálogo de usuarios |

## Objetivo

1. Cargar ratings, repos y usuarios
2. Explorar distribuciones (histograma de puntuaciones, conteos)
3. Construir la matriz usuario–ítem
4. Ajustar `NearestNeighbors` sobre vectores de usuario
5. Recomendar repos no valorados y evaluar error (MSE en hold-out)

## Dependencias

- Python 3
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`

```bash
source .venv/bin/activate
pip install -r src/ML/Recomendador/requirements.txt
```

## Cómo ejecutarlo

Desde la raíz del repositorio, con el venv activado:

```bash
# Pipeline completo (abre ventana de histograma)
python src/ML/Recomendador/recomendador.py

# Sin display gráfico
MPLBACKEND=Agg python -c "
import sys
sys.path.insert(0, 'src/ML/Recomendador')
from recomendador import runPipeline
runPipeline(showPlots=False)
"
```

### En local (Jupyter)

```bash
source .venv/bin/activate
pip install jupyter
cd src/ML/Recomendador
jupyter notebook Recomendador.ipynb
```

### En Google Colab

1. Abre el notebook en [Colab](https://colab.research.google.com/)
2. Sube los CSV de `data/` y el archivo `recomendador.py`
3. Ejecuta las celdas en orden

## Constantes relevantes

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `CSV_RATINGS_FILE` | `data/ratings.csv` | Ratings usuario–repo |
| `CSV_REPOS_FILE` | `data/repos.csv` | Metadatos de repos |
| `CSV_USERS_FILE` | `data/users.csv` | Metadatos de usuarios |
| `N_FIRST_ROWS` | `10` | Filas a mostrar en exploración |
| `DEFAULT_N_NEIGHBORS` | `5` | Vecinos para k-NN colaborativo |
| `DEFAULT_TOP_N` | `10` | Recomendaciones a devolver |
| `DEFAULT_TEST_SIZE` | `0.2` | Proporción hold-out para MSE |

## Datasets

### `ratings.csv`

| Columna | Significado |
|---------|-------------|
| `userId` | Identificador de usuario |
| `repoId` | Identificador de repositorio |
| `rating` | Puntuación (1–8) |

### `repos.csv`

| Columna | Significado |
|---------|-------------|
| `repoId` | Identificador |
| `title` | Nombre / owner del repo |
| `categories` | Categorías |
| `stars` | Estrellas (si disponible) |

### `users.csv`

| Columna | Significado |
|---------|-------------|
| `userId` | Identificador |
| `username` | Handle |
| `name` | Nombre |

El dataset incluye **30 usuarios**, **167 repos valorados** y **324 ratings**.

## Tests

```bash
source .venv/bin/activate
pip install -r src/ML/Recomendador/requirements.txt pytest
MPLBACKEND=Agg pytest tests/test_recomendador.py -v
```

Los tests no requieren display ni red; cargan los CSV locales.

## Salida esperada

- Resúmenes estadísticos de ratings, repos y users
- Histograma de puntuaciones
- Matriz usuario–ítem y usuarios similares
- Top-N recomendaciones para un usuario (p. ej. `userId=1`)
- MSE en hold-out sobre ratings reservados

## Notas

- La matriz usuario–ítem usa `0` donde no hay valoración.
- La similitud entre usuarios se calcula con distancia coseno.
- Ejemplo didáctico; no persiste el modelo ni resultados.
- En entornos sin display usa `MPLBACKEND=Agg`.

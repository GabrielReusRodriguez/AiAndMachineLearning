# Recomendador de repositorios GitHub

Notebook didáctico de **filtrado colaborativo** basado en vecinos cercanos
(`NearestNeighbors`) para recomendar repositorios a partir de ratings de usuarios.

## Ubicación

```text
src/ML/Recomendador/
├── data/
│   ├── ratings.csv
│   ├── repos.csv
│   └── users.csv
├── Recomendador.ipynb
└── requirements.txt
```

| Recurso | Propósito |
|---------|-----------|
| `Recomendador.ipynb` | Exploración de datos, matriz usuario–repo y recomendaciones k-NN |
| `data/ratings.csv` | Valoraciones usuario–repositorio |
| `data/repos.csv` | Catálogo de repositorios |
| `data/users.csv` | Catálogo de usuarios |

## Objetivo

1. Cargar ratings, repos y usuarios
2. Explorar distribuciones (histograma de puntuaciones, conteos)
3. Construir la matriz usuario–ítem
4. Ajustar `NearestNeighbors` y generar recomendaciones / evaluar error

## Dependencias

- Python 3
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`

```bash
source .venv/bin/activate
pip install -r src/ML/Recomendador/requirements.txt
```

## Cómo ejecutarlo

### En local (Jupyter)

```bash
source .venv/bin/activate
pip install jupyter
cd src/ML/Recomendador
jupyter notebook Recomendador.ipynb
```

### En Google Colab

1. Abre el notebook en [Colab](https://colab.research.google.com/)
2. Sube los CSV de `data/` o ajusta las rutas
3. Ejecuta las celdas en orden

## Constantes relevantes

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `CSV_RATINGS_FILE` | `./data/ratings.csv` | Ratings usuario–repo |
| `CSV_REPOS_FILE` | `./data/repos.csv` | Metadatos de repos |
| `CSV_USERS_FILE` | `./data/users.csv` | Metadatos de usuarios |
| `N_FIRST_ROWS` | `10` | Filas a mostrar en exploración |

## Datasets

### `ratings.csv`

| Columna | Significado |
|---------|-------------|
| `userId` | Identificador de usuario |
| `repoId` | Identificador de repositorio |
| `rating` | Puntuación |

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

## Salida esperada

- Resúmenes estadísticos de ratings, repos y users
- Histograma de puntuaciones
- Matriz de similitud / vecinos y recomendaciones para un usuario
- Métricas de error (p. ej. MSE) según las celdas de evaluación del notebook

## Notas

- Ejemplo didáctico de recomendación por vecinos; no persiste el modelo.
- En Colab hay que asegurar que los tres CSV estén accesibles respecto a las constantes de ruta.

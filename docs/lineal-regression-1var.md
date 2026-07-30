# Regresion lineal (1 variable)

Ejemplo didactico de **regresion lineal simple** para predecir el numero de
compartidos (`# Shares`) de un articulo a partir de su longitud (`Word count`).

## Ubicacion

```text
src/ML/LinealRegression1var/
├── data/
│   └── articulos_ml.csv
├── linealRegression1var.py
└── requirements.txt
```

| Recurso | Proposito |
|---------|-----------|
| `linealRegression1var.py` | Script ejecutable con funciones reutilizables (carga, filtro, fit, metricas, plot) |
| `data/articulos_ml.csv` | Dataset de articulos de ML con metricas de engagement |

## Objetivo

1. Cargar y filtrar el CSV de articulos
2. Entrenar una regresion lineal `Word count` → `# Shares`
3. Evaluar el modelo (MSE, R² / score)
4. Visualizar el scatter y la recta de ajuste

## Dependencias

- Python 3
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`

```bash
source .venv/bin/activate
pip install -r src/ML/LinealRegression1var/requirements.txt
```

## Como ejecutarlo

Desde la raiz del repositorio, con el venv activado:

```bash
# Pipeline completo (abre ventana de plot)
python src/ML/LinealRegression1var/linealRegression1var.py

# Sin display grafico
MPLBACKEND=Agg python -c "
import sys
sys.path.insert(0, 'src/ML/LinealRegression1var')
from linealRegression1var import runPipeline
runPipeline(showPlots=False)
"
```

## Dataset

| Columna | Significado |
|---------|-------------|
| `Title` | Titulo del articulo |
| `url` | URL (puede estar vacia) |
| `Word count` | Numero de palabras |
| `# of Links` | Enlaces en el articulo |
| `# of comments` | Comentarios |
| `# Images video` | Imagenes / videos |
| `Elapsed days` | Dias desde publicacion |
| `# Shares` | Veces compartido (target) |

Antes del entrenamiento se filtran filas con `Word count > 3500` o
`# Shares > 80000` para quedarse con la mayoria de los datos tipicos.

## Tests

```bash
source .venv/bin/activate
pip install -r src/ML/LinealRegression1var/requirements.txt pytest
MPLBACKEND=Agg pytest tests/test_lineal_regression_1var.py -v
```

Los tests no requieren display ni red; cargan `articulos_ml.csv` localmente.

## Salida esperada

- Conteo de filas originales vs filtradas en consola
- MSE, R² (variance score) y `score` del modelo
- Scatter coloreado por umbral de `Word count` y recta de regresion en rojo

## Notas

- Modelo de **una sola feature** (`Word count`); el R² suele ser bajo: la
  longitud del texto no explica bien por si sola los compartidos.
- Ejemplo didactico; no hace train/test split ni persiste el modelo.
- En entornos sin display usa `MPLBACKEND=Agg`.

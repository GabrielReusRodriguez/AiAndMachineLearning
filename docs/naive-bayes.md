# Clasificador Naive Bayes

Ejemplo didáctico que usa **Gaussian Naive Bayes** para recomendar si conviene
**comprar o alquilar** una vivienda según variables socioeconómicas.

## Ubicación

```text
src/ML/NaiveBayes/
├── data/
│   └── comprar_alquilar.csv
├── naiveBayes.py
├── naiveBayes.ipynb
└── requirements.txt
```

| Recurso | Propósito |
|---------|-----------|
| `naiveBayes.py` | Script ejecutable con funciones reutilizables (carga, features, fit, eval) |
| `naiveBayes.ipynb` | Notebook exploratorio equivalente (Colab / Jupyter) |
| `data/comprar_alquilar.csv` | Dataset etiquetado comprar (`1`) / alquilar (`0`) |

## Objetivo

1. Cargar y explorar el CSV de perfiles financieros
2. Crear columnas derivadas (`gastos`, `inversion`)
3. Seleccionar las `N_BEST_COLUMNS` features más informativas (`SelectKBest`)
4. Entrenar `GaussianNB` con split train/test
5. Evaluar con classification report y matriz de confusión

## Dependencias

- Python 3
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

```bash
source .venv/bin/activate
pip install -r src/ML/NaiveBayes/requirements.txt
```

## Cómo ejecutarlo

Desde la raíz del repositorio, con el venv activado:

```bash
# Pipeline completo (abre ventanas de plot)
python src/ML/NaiveBayes/naiveBayes.py

# Sin display gráfico
MPLBACKEND=Agg python -c "
from pathlib import Path
import sys
sys.path.insert(0, 'src/ML/NaiveBayes')
from naiveBayes import runPipeline
runPipeline(showPlots=False)
"
```

También puedes abrir `naiveBayes.ipynb` en Jupyter o [Google Colab](https://colab.research.google.com/).

## Constantes relevantes

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `CSV_FILE` | `data/comprar_alquilar.csv` | Ruta al dataset |
| `N_FIRST_ROWS` | `10` | Filas a mostrar en exploración |
| `N_BEST_COLUMNS` | `5` | Features seleccionadas con `SelectKBest` |
| `TEST_SIZE` | `0.2` | Proporción del conjunto de prueba |
| `RANDOM_STATE` | `6` | Semilla para el split train/test |

## Dataset

| Columna | Significado |
|---------|-------------|
| `ingresos` | Ingresos |
| `gastos_comunes` | Gastos comunes |
| `pago_coche` | Pago de coche |
| `gastos_otros` | Otros gastos |
| `ahorros` | Ahorros |
| `vivienda` | Precio / valor de la vivienda |
| `estado_civil` | Estado civil (categórico numérico) |
| `hijos` | Número de hijos |
| `trabajo` | Tipo / nivel de trabajo |
| `comprar` | Target: `1` comprar, `0` alquilar |

Columnas derivadas en el pipeline:

| Columna | Cálculo |
|---------|---------|
| `gastos` | `gastos_comunes + gastos_otros + pago_coche` |
| `inversion` | `vivienda - ahorros` |

Con `SelectKBest(k=5)` sobre el dataset completo, las features elegidas son:
`ingresos`, `ahorros`, `hijos`, `trabajo`, `inversion`.

## Tests

```bash
source .venv/bin/activate
pip install -r src/ML/NaiveBayes/requirements.txt pytest
MPLBACKEND=Agg pytest tests/test_naive_bayes.py -v
```

Los tests no requieren display ni red; cargan `comprar_alquilar.csv` localmente.

## Salida esperada

- Resumen estadístico y visualizaciones exploratorias
- Features seleccionadas por `SelectKBest`
- Precisión train ≈ 0.87 y test ≈ 0.90 (con `random_state=6`)
- Predicciones de ejemplo: perfiles `[2000, 5000, 0, 5, 200000]` → alquilar (`0`);
  `[6000, 34000, 2, 5, 320000]` → comprar (`1`)
- Classification report y matriz de confusión

## Notas

- Ejemplo didáctico basado en el teorema de Bayes; asume independencia condicional entre features.
- En Colab hay que asegurar que el CSV esté accesible respecto a `CSV_FILE`.
- En entornos sin display usa `MPLBACKEND=Agg`.

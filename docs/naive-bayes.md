# Clasificador Naive Bayes

Notebook didáctico que usa **Gaussian Naive Bayes** para recomendar si conviene
**comprar o alquilar** una vivienda según variables socioeconómicas.

## Ubicación

```text
src/ML/NaiveBayes/
├── data/
│   └── comprar_alquilar.csv
├── naiveBayes.ipynb
└── requirements.txt
```

| Recurso | Propósito |
|---------|-----------|
| `naiveBayes.ipynb` | Exploración, selección de features y clasificación |
| `data/comprar_alquilar.csv` | Dataset etiquetado comprar (`1`) / alquilar (`0`) |

## Objetivo

1. Cargar y explorar el CSV de perfiles financieros
2. Seleccionar las `N_BEST_COLUMNS` features más informativas (`SelectKBest`)
3. Entrenar `GaussianNB` con split train/test
4. Evaluar con classification report y matriz de confusión

## Dependencias

- Python 3
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

```bash
source .venv/bin/activate
pip install -r src/ML/NaiveBayes/requirements.txt
```

## Cómo ejecutarlo

### En local (Jupyter)

```bash
source .venv/bin/activate
pip install jupyter
cd src/ML/NaiveBayes
jupyter notebook naiveBayes.ipynb
```

### En Google Colab

1. Abre el notebook en [Colab](https://colab.research.google.com/)
2. Sube también `data/comprar_alquilar.csv` o ajusta `CSV_FILE`
3. Ejecuta las celdas en orden

## Constantes relevantes

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `CSV_FILE` | `./data/comprar_alquilar.csv` | Ruta al dataset |
| `N_FIRST_ROWS` | `10` | Filas a mostrar en exploración |
| `N_BEST_COLUMNS` | `5` | Features seleccionadas con `SelectKBest` |

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

## Salida esperada

- Resumen estadístico y visualizaciones exploratorias
- Features seleccionadas por `SelectKBest`
- Predicciones, classification report y matriz de confusión

## Notas

- Ejemplo didáctico basado en el teorema de Bayes; asume independencia condicional entre features.
- En Colab hay que asegurar que el CSV esté accesible respecto a `CSV_FILE`.

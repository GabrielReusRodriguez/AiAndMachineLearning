# Regresión lineal con 2 variables

Ejemplo didáctico de **regresión lineal múltiple** para predecir el número de
shares de artículos de Machine Learning a partir del conteo de palabras y una
métrica agregada de engagement (enlaces + comentarios + imágenes).

## Ubicación

```text
src/ML/LinearRegression2var/
├── data/
│   └── articulos_ml.csv
├── linealRegression2var.py
└── requirements.txt
```

| Recurso | Propósito |
|---------|-----------|
| `linealRegression2var.py` | Módulo con funciones reutilizables y `runPipeline()` |
| `data/articulos_ml.csv` | Dataset de artículos con métricas de contenido y shares |

## Objetivo

1. Cargar y filtrar el CSV (word count ≤ 3500, shares ≤ 80000)
2. Construir la feature `suma` = enlaces + comentarios + imágenes
3. Entrenar `LinearRegression` con `Word count` y `suma` para predecir `# Shares`
4. Reportar MSE, R² y una predicción de ejemplo; visualizar el plano en 3D

## API principal

| Función | Descripción |
|---------|-------------|
| `loadArticlesData()` | Carga el CSV desde `data/articulos_ml.csv` |
| `filterArticles()` | Filtra por word count y shares máximos |
| `buildEngagementSum()` | Suma enlaces, comentarios (NaN→0) e imágenes |
| `buildFeatureFrame()` | DataFrame con `Word count` y `suma` |
| `fitLinearRegression()` | Entrena el modelo sklearn |
| `evaluateRegression()` | Devuelve MSE y R² |
| `predictShares()` | Predicción para un par (palabras, engagement) |
| `plotRegression3D()` | Nube 3D + plano de regresión |
| `runPipeline()` | Orquesta carga, entrenamiento, métricas y gráfico |

## Dependencias

- Python 3
- `pandas`, `numpy`, `matplotlib`, `scikit-learn`

```bash
source .venv/bin/activate
pip install -r src/ML/LinearRegression2var/requirements.txt
```

## Cómo ejecutarlo

El módulo resuelve la ruta del CSV respecto a su propia ubicación, así que puede
ejecutarse desde cualquier directorio:

```bash
source .venv/bin/activate
python src/ML/LinearRegression2var/linealRegression2var.py
```

Sin display gráfico:

```bash
MPLBACKEND=Agg python src/ML/LinearRegression2var/linealRegression2var.py
```

Uso programático:

```python
from linealRegression2var import runPipeline

result = runPipeline(showPlots=False)
print(result["metrics"])
print(result["examplePrediction"])
```

## Tests

```bash
source .venv/bin/activate
MPLBACKEND=Agg pytest tests/test_linear_regression_2var.py -v
```

Los tests cubren filtrado, construcción de features, entrenamiento sintético,
predicción escalar y el pipeline completo sobre el CSV real (sin abrir ventanas
gráficas).

## Dataset

| Columna | Significado |
|---------|-------------|
| `Title` | Título del artículo |
| `url` | URL |
| `Word count` | Número de palabras |
| `# of Links` | Enlaces en el artículo |
| `# of comments` | Comentarios |
| `# Images video` | Imágenes / vídeo |
| `Elapsed days` | Días desde publicación |
| `# Shares` | Veces compartido (target) |

## Salida esperada

- Coeficientes del modelo (2 features)
- Mean squared error y variance score (R²)
- Predicción de shares para un ejemplo (`Word count=2000`, engagement `10+4+6`)
- Gráfico 3D con nube de puntos y plano de regresión

Valores de referencia aproximados sobre el CSV filtrado:

```text
Coeficientes: [6.63, -483.41]
Mean squared error: 352122816.48
Variance score (R²): 0.11
```

## Notas

- Ejemplo didáctico; entrena y evalúa sobre el mismo conjunto filtrado (sin split train/test).
- En entornos sin display usa `MPLBACKEND=Agg` o `runPipeline(showPlots=False)`.
- La feature `suma` agrega señales de engagement; un coeficiente negativo puede indicar multicolinealidad o ruido en datos reales.

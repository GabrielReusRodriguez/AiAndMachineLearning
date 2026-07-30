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
| `linealRegression2var.py` | Script ejecutable: filtro, entrenamiento, métricas y plano 3D |
| `data/articulos_ml.csv` | Dataset de artículos con métricas de contenido y shares |

## Objetivo

1. Cargar y filtrar el CSV (word count ≤ 3500, shares ≤ 80000)
2. Construir la feature `suma` = enlaces + comentarios + imágenes
3. Entrenar `LinearRegression` con `Word count` y `suma` para predecir `# Shares`
4. Reportar MSE, R² y una predicción de ejemplo; visualizar el plano en 3D

## Dependencias

- Python 3
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

```bash
source .venv/bin/activate
pip install -r src/ML/LinearRegression2var/requirements.txt
```

## Cómo ejecutarlo

El script usa rutas relativas a `./data/`. Ejecútalo desde la carpeta del ejemplo:

```bash
source .venv/bin/activate
cd src/ML/LinearRegression2var
python linealRegression2var.py
```

Sin display gráfico:

```bash
MPLBACKEND=Agg python linealRegression2var.py
```

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

- Coeficientes del modelo
- Mean squared error y variance score (R²)
- Predicción de shares para un ejemplo (`Word count=2000`, engagement `10+4+6`)
- Gráfico 3D con nube de puntos y plano de regresión

## Notas

- Ejemplo didáctico; entrena y evalúa sobre el mismo conjunto filtrado (sin split train/test).
- En entornos sin display usa `MPLBACKEND=Agg`.

# Regresión logística

Ejemplos de **clasificación** con regresión logística: uno sobre usuarios web
(sistema operativo) y otro con datos sintéticos y frontera de decisión.

## Ubicación

```text
src/ML/RegresionLogistica/
├── data/
│   └── usuarios_win_mac_lin.csv
├── regresionLogistica.py
├── regresionLogistica_v2.py
└── requirements.txt
```

| Recurso | Propósito |
|---------|-----------|
| `regresionLogistica.py` | Clasificación multiclase Windows / Mac / Linux |
| `regresionLogistica_v2.py` | Clasificación binaria con datos sintéticos y visualización |
| `data/usuarios_win_mac_lin.csv` | Dataset de comportamiento web etiquetado por SO |

## Objetivo

### `regresionLogistica.py`

1. Explorar el CSV (`duracion`, `paginas`, `acciones`, `valor`, `clase`)
2. Entrenar `LogisticRegression` para predecir el SO (`0` Windows, `1` Mac, `2` Linux)
3. Validar con hold-out (20 %) y validación cruzada K-Fold
4. Mostrar accuracy, matriz de confusión y classification report

### `regresionLogistica_v2.py`

1. Generar datos sintéticos 2D con etiqueta binaria
2. Entrenar y evaluar el modelo
3. Dibujar la frontera de decisión

## Dependencias

- Python 3
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

```bash
source .venv/bin/activate
pip install -r src/ML/RegresionLogistica/requirements.txt
```

## Cómo ejecutarlo

Los scripts usan rutas relativas a `./data/` (solo el primero). Ejecuta desde la
carpeta del ejemplo:

```bash
source .venv/bin/activate
cd src/ML/RegresionLogistica

python regresionLogistica.py
python regresionLogistica_v2.py
```

Sin display:

```bash
MPLBACKEND=Agg python regresionLogistica.py
MPLBACKEND=Agg python regresionLogistica_v2.py
```

## Dataset (`usuarios_win_mac_lin.csv`)

| Columna | Significado |
|---------|-------------|
| `duracion` | Duración de la sesión |
| `paginas` | Páginas visitadas |
| `acciones` | Acciones realizadas |
| `valor` | Valor asociado a la sesión |
| `clase` | SO: `0` Windows, `1` Mac, `2` Linux |

## Salida esperada

- Resumen estadístico y conteo por clase
- Histogramas / pairplot (script v1)
- Score del modelo, CV accuracy, confusion matrix y classification report
- Frontera de decisión 2D (script v2)

## Notas

- Ejemplo didáctico; no persiste el modelo.
- En entornos sin display usa `MPLBACKEND=Agg`.

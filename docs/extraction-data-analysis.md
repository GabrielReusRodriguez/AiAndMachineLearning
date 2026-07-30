# Extraccion y analisis de datos

Ejemplos en Python que cargan datasets publicos de paises, exploran estadisticas,
calculan correlaciones, detectan outliers y comparan la evolucion de poblacion.

## Ubicacion

```text
src/ML/ExtractionDataAnalysis/
├── read_countries.py
├── read_countries_correlation.py
├── outliers_detection.py
├── compare_population_of_countries.py
└── requirements.txt
```

| Script | Proposito |
|--------|-----------|
| `read_countries.py` | Carga el CSV de paises y muestra shape, info y describe |
| `read_countries_correlation.py` | Matriz de correlacion numerica (plot con statsmodels) |
| `outliers_detection.py` | Filtra paises de habla hispana y elimina outliers (mean +/- 2*std) |
| `compare_population_of_countries.py` | Compara la poblacion de dos paises a lo largo del tiempo |

## Funciones exportadas

| Modulo | Funcion | Descripcion |
|--------|---------|-------------|
| `read_countries` | `loadCountries(url)` | Lee el CSV remoto con separador `;` e indice `alpha_3` |
| `read_countries` | `summarizeCountries(df)` | Devuelve `shape` y `describe()` del DataFrame |
| `read_countries_correlation` | `computeCorrelation(df)` | Matriz de correlacion solo con columnas numericas |
| `read_countries_correlation` | `plotCorrelation(corr)` | Grafico de la matriz (statsmodels + matplotlib) |
| `outliers_detection` | `filterSpanishSpeaking(df)` | Filtra filas cuyo campo `languages` contiene `"es"` |
| `outliers_detection` | `findOutliers(data)` | Indices fuera de mean +/- 2*std en la primera columna numerica |
| `outliers_detection` | `removeOutliers(df, indices)` | Elimina filas por indice |
| `compare_population_of_countries` | `loadPopulation(url)` | Lee el CSV de evolucion de poblacion |
| `compare_population_of_countries` | `buildComparisonFrame(df, c1, c2)` | DataFrame indexado por anio con ambos paises |
| `compare_population_of_countries` | `compare2Countries(c1, c2)` | Orquesta carga, comparacion y grafico de barras |

## Objetivo

1. Leer datasets CSV remotos con pandas
2. Explorar resumenes estadisticos y correlaciones
3. Detectar y eliminar anomalias con el criterio mean +/- 2 desviaciones tipicas
4. Visualizar la evolucion de poblacion de dos paises

## Dependencias

- Python 3
- `pandas`, `numpy`, `matplotlib`, `statsmodels`

```bash
source .venv/bin/activate
pip install -r src/ML/ExtractionDataAnalysis/requirements.txt
```

## Como ejecutarlo

Desde la raiz del repositorio, con el venv activado:

```bash
# Resumen del dataset de paises
python src/ML/ExtractionDataAnalysis/read_countries.py

# Matriz de correlacion (abre una ventana de plot)
python src/ML/ExtractionDataAnalysis/read_countries_correlation.py

# Outliers en paises de habla hispana
python src/ML/ExtractionDataAnalysis/outliers_detection.py

# Comparar poblacion (nombres de pais, no codigos ISO)
python src/ML/ExtractionDataAnalysis/compare_population_of_countries.py Spain France
```

## Fuentes de datos

| Constante / URL | Dataset |
|-----------------|---------|
| `COUNTRIES_URL` | [list-of-countries](https://raw.githubusercontent.com/lorey/list-of-countries/master/csv/countries.csv) (separador `;`, indice `alpha_3`) |
| `POPULATION_URL` | [Population_Growth](https://raw.githubusercontent.com/DrueStaples/Population_Growth/master/countries.csv) |

## Tests

```bash
source .venv/bin/activate
pip install -r src/ML/ExtractionDataAnalysis/requirements.txt pytest
MPLBACKEND=Agg pytest tests/test_extraction_data_analysis.py -v
```

Los tests unitarios no requieren red ni abren ventanas graficas (`MPLBACKEND=Agg`). Los smoke de
carga remota se omiten automaticamente si no hay acceso a `COUNTRIES_URL`.

Cobertura principal:

- `findOutliers`, `filterSpanishSpeaking`, `removeOutliers` con DataFrames en memoria
- `buildComparisonFrame` y validacion de pais invalido
- `summarizeCountries` y `loadPopulation` con CSV local
- Smoke de `loadCountries` y `computeCorrelation` contra la URL publica

## Salida esperada

- `read_countries.py`: shape, `info()` y `describe()` por consola
- `read_countries_correlation.py`: grafico de la matriz de correlacion
- `outliers_detection.py`: grafico de barras population/area sin outliers
- `compare_population_of_countries.py`: grafico de barras comparando dos paises

## Notas

- `compare_population_of_countries.py` espera **nombres de pais** (`Spain`, `France`), no codigos `alpha_3`.
- Los scripts de plot llaman a `plt.show()`; en entornos sin display usa `MPLBACKEND=Agg`.
- Ejemplo didactico de exploracion de datos; no persiste modelos ni resultados.

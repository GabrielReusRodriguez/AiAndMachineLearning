# Extraccion y analisis de datos

Ejemplos en Python que cargan datasets publicos de paises, exploran estadisticas,
calculan correlaciones, detectan outliers y comparan la evolucion de poblacion.

## Ubicacion

```text
src/ML/ExtractionDataAnalysis/
```

| Script | Proposito |
|--------|-----------|
| `read_countries.py` | Carga el CSV de paises y muestra shape, info y describe |
| `read_countries_correlation.py` | Matriz de correlacion numerica (plot con statsmodels) |
| `outliers_detection.py` | Filtra paises de habla hispana y elimina outliers (mean +/- 2*std) |
| `compare_population_of_countries.py` | Compara la poblacion de dos paises a lo largo del tiempo |

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

Los tests unitarios no requieren red. Los smoke de carga remota se omiten si no hay acceso a `COUNTRIES_URL`.

## Salida esperada

- `read_countries.py`: shape, `info()` y `describe()` por consola
- `read_countries_correlation.py`: grafico de la matriz de correlacion
- `outliers_detection.py`: grafico de barras population/area sin outliers
- `compare_population_of_countries.py`: grafico de barras comparando dos paises

## Notas

- `compare_population_of_countries.py` espera **nombres de pais** (`Spain`, `France`), no codigos `alpha_3`.
- Los scripts de plot llaman a `plt.show()`; en entornos sin display usa `MPLBACKEND=Agg`.
- Ejemplo didactico de exploracion de datos; no persiste modelos ni resultados.

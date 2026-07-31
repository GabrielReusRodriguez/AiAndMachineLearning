# Configuración del entorno

## Python y entorno virtual

El proyecto usa Python 3 con un entorno virtual local en la raíz del repositorio:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

Mantén el entorno activado al instalar dependencias y al ejecutar código o tests.

## Dependencias

Las dependencias varían según el ejemplo. Cada carpeta bajo `src/AI/` o `src/ML/` incluye su propio `requirements.txt` (salvo notebooks que solo documentan `pip install` en la guía). Instálalas solo cuando vayas a ejecutar ese ejemplo:

```bash
source .venv/bin/activate
pip install -r src/ML/K-means/requirements.txt

# Ejemplo AI standalone (TensorFlow)
pip install -r src/AI/StandAlone/FirstNeuralTensorFlow/requirements.txt
```

Consulta la guía correspondiente en esta carpeta `docs/`.

## Scripts standalone vs Colab

| Ubicación | Uso típico |
|-----------|------------|
| `src/AI/Colab/` | Notebooks pensados para Google Colab / Jupyter |
| `src/AI/StandAlone/` | Scripts `.py` ejecutables en local con el venv |
| `src/ML/` | Ejemplos de Machine Learning (script y/o notebook) |

## Google Colab

Los notebooks bajo `src/AI/Colab/` y `src/ML/` están pensados para ejecutarse también en Google Colab:

1. Abre [Google Colab](https://colab.research.google.com/)
2. Sube el `.ipynb` o ábrelo desde GitHub / Drive
3. Ejecuta las celdas en orden

Algunos scripts `.py` usan rutas relativas a `./data/`; en local ejecútalos desde la carpeta del ejemplo o ajusta la ruta.

## Google Coral TPU (opcional)

Para inferencia con acelerador USB:

1. Conecta el dispositivo Coral TPU
2. Instala el runtime Edge TPU según la [documentación oficial](https://coral.ai/docs/)
3. Usa los ejemplos del proyecto que indiquen soporte para Coral

## Tests

Con el venv activado y las dependencias del ejemplo instaladas:

```bash
MPLBACKEND=Agg pytest tests/ -v
```

`MPLBACKEND=Agg` evita abrir ventanas gráficas en entornos sin display.

## Verificación rápida

```bash
python3 --version
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
```

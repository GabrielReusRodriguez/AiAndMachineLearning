# Configuración del entorno

## Python y entorno virtual

El proyecto usa Python 3 con un entorno virtual local:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows
```

Mantén el entorno activado al instalar dependencias y al ejecutar código o tests.

## Dependencias

Las dependencias varían según el ejemplo. Instálalas solo cuando vayas a ejecutar ese ejemplo. Consulta la guía correspondiente en esta carpeta `docs/`.

## Google Colab

Los notebooks bajo `src/Colab/` están pensados para ejecutarse también en Google Colab:

1. Abre [Google Colab](https://colab.research.google.com/)
2. Sube el `.ipynb` o ábrelo desde GitHub / Drive
3. Ejecuta las celdas en orden

## Google Coral TPU (opcional)

Para inferencia con acelerador USB:

1. Conecta el dispositivo Coral TPU
2. Instala el runtime Edge TPU según la [documentación oficial](https://coral.ai/docs/)
3. Usa los ejemplos del proyecto que indiquen soporte para Coral

## Verificación rápida

```bash
python3 --version
source .venv/bin/activate
python -c "import sys; print(sys.executable)"
```

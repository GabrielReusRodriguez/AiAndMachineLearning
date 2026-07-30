# Tokenizador simple

Ejemplo en notebook que construye un vocabulario a partir de un texto y obtiene los índices de cada palabra con el `Tokenizer` de Keras.

## Ubicación

```text
src/AI/Colab/Simple_Tokenizer/SimpleTokenizer.ipynb
```

## Objetivo

Dado un texto de entrada (fragmento de *Don Quijote*), el notebook:

1. Divide el texto en frases
2. Ajusta un `Tokenizer` con un máximo de palabras (`MAX_NUM_WORDS = 1000`)
3. Genera el `word_index` (palabra → id)
4. Construye el diccionario inverso (id → palabra)

## Dependencias

- Python 3
- TensorFlow / Keras (`tensorflow.keras.preprocessing.text.Tokenizer`)

En un entorno local con venv activado:

```bash
pip install tensorflow
```

En Google Colab, TensorFlow suele estar preinstalado.

## Cómo ejecutarlo

### En Google Colab

1. Abre el notebook en Colab
2. Ejecuta las celdas en orden

### En local

```bash
source .venv/bin/activate
pip install tensorflow jupyter
jupyter notebook src/AI/Colab/Simple_Tokenizer/SimpleTokenizer.ipynb
```

## Constantes relevantes

| Constante | Valor | Significado |
|-----------|-------|-------------|
| `MAX_NUM_WORDS` | `1000` | Máximo de palabras en el vocabulario del tokenizer |

## Salida esperada

- `word_index`: diccionario ordenado por frecuencia (las palabras más frecuentes tienen índices más bajos)
- `reverse_word_index`: mapa inverso para recuperar el literal a partir del id

## Notas

- El tokenizer de Keras normaliza el texto (minúsculas, elimina puntuación) antes de construir el vocabulario.
- Este ejemplo es didáctico: no persiste el tokenizer ni convierte secuencias a padding; solo ilustra el ajuste del vocabulario.

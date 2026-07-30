"""Carga y resume el dataset de paises (list-of-countries)."""

import pandas as pd

COUNTRIES_URL = (
  "https://raw.githubusercontent.com/lorey/list-of-countries/master/csv/countries.csv"
)


def loadCountries(url: str = COUNTRIES_URL) -> pd.DataFrame:
  """Lee el CSV de paises con separador ';' e indice alpha_3."""
  return pd.read_csv(url, sep=";", index_col="alpha_3")


def summarizeCountries(dataFrame: pd.DataFrame) -> dict:
  """Devuelve shape y describe() del DataFrame."""
  return {
    "shape": dataFrame.shape,
    "describe": dataFrame.describe(),
  }


if __name__ == "__main__":
  dataFrame = loadCountries()
  summary = summarizeCountries(dataFrame)
  print(f"Cantidad de Filas y columnas: {summary['shape']}")
  print("Informacion del DataFrame:")
  dataFrame.info()
  print()
  print(f"Resume: {summary['describe']}")

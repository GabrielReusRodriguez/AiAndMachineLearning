"""Compara la evolucion de poblacion de dos paises."""

import sys

import matplotlib.pyplot as plt
import pandas as pd

POPULATION_URL = (
  "https://raw.githubusercontent.com/DrueStaples/Population_Growth/master/countries.csv"
)


def loadPopulation(url: str = POPULATION_URL) -> pd.DataFrame:
  """Lee el CSV de poblacion por pais y anio."""
  return pd.read_csv(url)


def buildComparisonFrame(
  dataFrame: pd.DataFrame, country1: str, country2: str
) -> pd.DataFrame:
  """Construye un DataFrame indexado por anio con la poblacion de ambos paises."""
  if country1 not in dataFrame["country"].values:
    raise ValueError(f"Error, {country1} is not a valid country name.")
  if country2 not in dataFrame["country"].values:
    raise ValueError(f"Error, {country2} is not a valid country name.")

  dfCountry1 = dataFrame[dataFrame["country"] == country1]
  dfCountry2 = dataFrame[dataFrame["country"] == country2]
  anios = dfCountry1["year"].unique()
  dfCountry1 = dfCountry1.drop(["country"], axis="columns")
  dfCountry2 = dfCountry2.drop(["country"], axis="columns")
  return pd.DataFrame(
    {
      country1: dfCountry1["population"].values,
      country2: dfCountry2["population"].values,
    },
    index=anios,
  )


def compare2Countries(country1: str, country2: str, url: str = POPULATION_URL) -> int:
  """Carga datos, compara dos paises y muestra un grafico de barras. Devuelve 0/1."""
  dataFrame = loadPopulation(url)
  try:
    dfPlot = buildComparisonFrame(dataFrame, country1, country2)
  except ValueError as err:
    print(err)
    return 1
  dfPlot.plot(kind="bar")
  plt.show()
  return 0


if __name__ == "__main__":
  if len(sys.argv) < 3:
    print("ERROR: this program requires two country names to compare")
    sys.exit(1)
  sys.exit(compare2Countries(sys.argv[1], sys.argv[2]))

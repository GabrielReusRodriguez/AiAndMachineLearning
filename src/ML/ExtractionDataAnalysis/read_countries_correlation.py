"""Matriz de correlacion numerica del dataset de paises."""

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm

from read_countries import COUNTRIES_URL, loadCountries


def computeCorrelation(dataFrame: pd.DataFrame) -> pd.DataFrame:
  """Calcula la matriz de correlacion solo con columnas numericas."""
  return dataFrame.corr(numeric_only=True)


def plotCorrelation(corr: pd.DataFrame) -> None:
  """Dibuja la matriz de correlacion con statsmodels."""
  sm.graphics.plot_corr(corr, xnames=list(corr.columns))
  plt.show()


if __name__ == "__main__":
  dataFrame = loadCountries(COUNTRIES_URL)
  corr = computeCorrelation(dataFrame)
  plotCorrelation(corr)

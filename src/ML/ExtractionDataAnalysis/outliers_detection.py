"""Deteccion de outliers (mean +/- 2*std) en paises de habla hispana."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from read_countries import COUNTRIES_URL, loadCountries


def filterSpanishSpeaking(dataFrame: pd.DataFrame) -> pd.DataFrame:
  """Filtra paises cuyo campo languages contiene 'es'."""
  cleaned = dataFrame.replace(np.nan, "", regex=True)
  return cleaned[cleaned["languages"].str.contains("es")]


def findOutliers(data: pd.DataFrame) -> list:
  """Indices de filas con algun valor fuera de mean +/- 2*std (columna 0)."""
  anomalies = []
  dataSd = data.std()
  dataMean = data.mean()
  lowerLimit = dataMean - dataSd * 2
  upperLimit = dataMean + dataSd * 2

  for index, row in data.iterrows():
    if row.iloc[0] > upperLimit.iloc[0] or row.iloc[0] < lowerLimit.iloc[0]:
      anomalies.append(index)
  return anomalies


def removeOutliers(dataFrame: pd.DataFrame, anomalies: list) -> pd.DataFrame:
  """Elimina las filas cuyo indice esta en anomalies."""
  return dataFrame.drop(anomalies)


if __name__ == "__main__":
  dataFrame = loadCountries(COUNTRIES_URL)
  dfEspanol = filterSpanishSpeaking(dataFrame)
  dataNumeric = dfEspanol.select_dtypes(include=np.number)
  anomalies = findOutliers(dataNumeric)
  cleanDataFrame = removeOutliers(dfEspanol, anomalies)
  cleanDataFrame[["population", "area"]].sort_values(["population"]).plot(
    kind="bar", rot=65, figsize=(20, 10)
  )
  plt.show()

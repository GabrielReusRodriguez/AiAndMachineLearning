"""Regresion lineal simple: Word count -> # Shares (articulos_ml.csv)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_FILE = DATA_DIR / "articulos_ml.csv"
FEATURE_COLUMN = "Word count"
TARGET_COLUMN = "# Shares"
MAX_WORD_COUNT = 3500
MAX_SHARES = 80000
WORD_COUNT_COLOR_THRESHOLD = 1808
SCATTER_COLORS = ["orange", "blue"]
SCATTER_SIZE = 30


def loadArticlesData(csvPath: Path | str = CSV_FILE) -> pd.DataFrame:
  """Carga el CSV de articulos de Machine Learning."""
  return pd.read_csv(csvPath)


def filterArticles(
  dataFrame: pd.DataFrame,
  maxWordCount: int = MAX_WORD_COUNT,
  maxShares: int = MAX_SHARES,
) -> pd.DataFrame:
  """Filtra articulos por limites de Word count y # Shares."""
  if maxWordCount < 0 or maxShares < 0:
    raise ValueError("maxWordCount y maxShares deben ser >= 0")
  return dataFrame[
    (dataFrame[FEATURE_COLUMN] <= maxWordCount)
    & (dataFrame[TARGET_COLUMN] <= maxShares)
  ].copy()


def extractFeatureAndTarget(
  dataFrame: pd.DataFrame,
  featureColumn: str = FEATURE_COLUMN,
  targetColumn: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Extrae X (feature) e y (target) como DataFrames de una columna."""
  return dataFrame[[featureColumn]], dataFrame[[targetColumn]]


def trainLinearRegression(
  features: pd.DataFrame | np.ndarray,
  target: pd.DataFrame | np.ndarray,
) -> linear_model.LinearRegression:
  """Ajusta un modelo de regresion lineal y lo devuelve entrenado."""
  model = linear_model.LinearRegression()
  return model.fit(features, target)


def predictShares(
  model: linear_model.LinearRegression,
  features: pd.DataFrame | np.ndarray,
) -> np.ndarray:
  """Predice # Shares a partir de Word count."""
  return model.predict(features)


def evaluateModel(
  targetTrue: pd.DataFrame | np.ndarray,
  targetPred: np.ndarray,
  model: linear_model.LinearRegression | None = None,
  features: pd.DataFrame | np.ndarray | None = None,
) -> dict[str, float]:
  """Calcula MSE, R2 y opcionalmente score del modelo."""
  metrics = {
    "meanSquaredError": float(mean_squared_error(targetTrue, targetPred)),
    "r2Score": float(r2_score(targetTrue, targetPred)),
  }
  if model is not None and features is not None:
    metrics["score"] = float(model.score(features, targetTrue))
  return metrics


def buildRegressionLine(
  model: linear_model.LinearRegression,
  xMin: float,
  xMax: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Construye puntos (x, y) de la recta de regresion entre xMin y xMax."""
  if xMax < xMin:
    raise ValueError("xMax debe ser >= xMin")
  slope = float(np.asarray(model.coef_).reshape(-1)[0])
  intercept = float(np.asarray(model.intercept_).reshape(-1)[0])
  xLine = np.arange(int(xMin), int(xMax) + 1)
  yLine = slope * xLine + intercept
  return xLine, yLine


def colorByWordCount(
  dataFrame: pd.DataFrame,
  threshold: int = WORD_COUNT_COLOR_THRESHOLD,
  colors: list[str] | None = None,
) -> list[str]:
  """Asigna color por umbral de Word count (naranja si supera threshold)."""
  palette = colors or SCATTER_COLORS
  return [
    palette[0] if row[FEATURE_COLUMN] > threshold else palette[1]
    for _, row in dataFrame.iterrows()
  ]


def plotScatterWithRegression(
  dataFrame: pd.DataFrame,
  model: linear_model.LinearRegression,
  show: bool = True,
) -> plt.Figure:
  """Scatter Word count vs # Shares con la recta de regresion."""
  wordCounts = dataFrame[FEATURE_COLUMN].values
  shares = dataFrame[TARGET_COLUMN].values
  pointColors = colorByWordCount(dataFrame)
  xLine, yLine = buildRegressionLine(
    model,
    float(wordCounts.min()),
    float(wordCounts.max()),
  )

  fig, ax = plt.subplots()
  ax.scatter(wordCounts, shares, c=pointColors, s=SCATTER_SIZE)
  ax.plot(xLine, yLine, color="red")
  ax.set_xlabel(FEATURE_COLUMN)
  ax.set_ylabel(TARGET_COLUMN)
  ax.set_title("Regresion lineal: Word count vs # Shares")
  if show:
    plt.show()
  return fig


def runPipeline(showPlots: bool = True) -> dict:
  """Ejecuta carga, filtrado, entrenamiento y evaluacion."""
  dataFrame = loadArticlesData()
  filtered = filterArticles(dataFrame)
  print(f"Filas originales: {len(dataFrame)}")
  print(f"Filas filtradas: {len(filtered)}")

  features, target = extractFeatureAndTarget(filtered)
  model = trainLinearRegression(features, target)
  predictions = predictShares(model, features)
  metrics = evaluateModel(target, predictions, model=model, features=features)

  print(f"Mean squared_error: {metrics['meanSquaredError']:.2f}")
  print(f"Variance score (R2): {metrics['r2Score']:.2f}")
  print(f"Score: {metrics['score']}")

  if showPlots:
    plotScatterWithRegression(filtered, model, show=True)

  return {
    "dataFrame": dataFrame,
    "filtered": filtered,
    "features": features,
    "target": target,
    "model": model,
    "predictions": predictions,
    "metrics": metrics,
  }


if __name__ == "__main__":
  runPipeline(showPlots=True)

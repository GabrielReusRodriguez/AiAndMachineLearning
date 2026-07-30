"""Regresión lineal múltiple: shares vs word count y engagement agregado."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_FILE = DATA_DIR / "articulos_ml.csv"
MAX_WORD_COUNT = 3500
MAX_SHARES = 80000
FEATURE_COLUMNS = ["Word count", "suma"]
TARGET_COLUMN = "# Shares"
LINKS_COLUMN = "# of Links"
COMMENTS_COLUMN = "# of comments"
IMAGES_COLUMN = "# Images video"
NUM_RESUME_LINES = 10
EXAMPLE_WORD_COUNT = 2000
EXAMPLE_ENGAGEMENT = 10 + 4 + 6


def loadArticlesData(csvPath: Path | str = CSV_FILE) -> pd.DataFrame:
  """Carga el CSV de artículos de Machine Learning."""
  return pd.read_csv(csvPath)


def filterArticles(
  dataFrame: pd.DataFrame,
  maxWordCount: int = MAX_WORD_COUNT,
  maxShares: int = MAX_SHARES,
) -> pd.DataFrame:
  """Filtra filas por word count y shares dentro de los límites."""
  return dataFrame[
    (dataFrame["Word count"] <= maxWordCount)
    & (dataFrame[TARGET_COLUMN] <= maxShares)
  ].copy()


def buildEngagementSum(dataFrame: pd.DataFrame) -> pd.Series:
  """Suma enlaces, comentarios (NaN→0) e imágenes/vídeo."""
  comments = dataFrame[COMMENTS_COLUMN].fillna(0)
  return dataFrame[LINKS_COLUMN] + comments + dataFrame[IMAGES_COLUMN]


def buildFeatureFrame(filteredDataFrame: pd.DataFrame) -> pd.DataFrame:
  """Construye el DataFrame de features: Word count y suma de engagement."""
  features = pd.DataFrame()
  features["Word count"] = filteredDataFrame["Word count"]
  features["suma"] = buildEngagementSum(filteredDataFrame)
  return features


def extractTrainingArrays(
  featureFrame: pd.DataFrame,
  targetSeries: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
  """Devuelve (X, y) como arrays numpy para el entrenamiento."""
  return np.array(featureFrame), np.array(targetSeries)


def fitLinearRegression(
  features: np.ndarray,
  targets: np.ndarray,
) -> linear_model.LinearRegression:
  """Entrena y devuelve un modelo de regresión lineal."""
  model = linear_model.LinearRegression()
  model.fit(features, targets)
  return model


def evaluateRegression(
  model: linear_model.LinearRegression,
  features: np.ndarray,
  targets: np.ndarray,
) -> dict[str, float]:
  """Calcula MSE y R² sobre las predicciones del modelo."""
  predictions = model.predict(features)
  return {
    "mse": mean_squared_error(targets, predictions),
    "r2": r2_score(targets, predictions),
  }


def predictShares(
  model: linear_model.LinearRegression,
  wordCount: float,
  engagementSum: float,
) -> float:
  """Predice shares para un par (word count, engagement)."""
  sample = np.array([[wordCount, engagementSum]])
  return float(model.predict(sample)[0])


def plotRegression3D(
  model: linear_model.LinearRegression,
  features: np.ndarray,
  targets: np.ndarray,
  show: bool = True,
) -> plt.Figure:
  """Gráfico 3D: nube de puntos y plano de regresión."""
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")

  xx, yy = np.meshgrid(
    np.linspace(0, MAX_WORD_COUNT, num=10),
    np.linspace(0, 60, num=10),
  )
  zz = model.coef_[0] * xx + model.coef_[1] * yy + model.intercept_
  ax.plot_surface(xx, yy, zz, alpha=0.2, cmap="hot")
  ax.scatter(features[:, 0], features[:, 1], targets, c="blue", s=30)
  ax.view_init(elev=30.0, azim=65)
  ax.set_xlabel("Cantidad de Palabras")
  ax.set_ylabel("Cantidad de enlaces, comentarios e imagenes")
  ax.set_zlabel("Compartido en redes")
  ax.set_title("Regresion lineal con multiples variables")
  if show:
    plt.show()
  return fig


def runPipeline(showPlots: bool = True) -> dict:
  """Ejecuta carga, filtrado, entrenamiento, métricas y visualización."""
  dataFrame = loadArticlesData()
  filtered = filterArticles(dataFrame)
  featureFrame = buildFeatureFrame(filtered)
  targets = filtered[TARGET_COLUMN].values
  features, targetArray = extractTrainingArrays(featureFrame, targets)

  model = fitLinearRegression(features, targetArray)
  metrics = evaluateRegression(model, features, targetArray)
  examplePrediction = predictShares(
    model,
    EXAMPLE_WORD_COUNT,
    EXAMPLE_ENGAGEMENT,
  )

  print(f"Coeficientes: \n {model.coef_}")
  print(f"Mean squared error : \n {metrics['mse']:.2f}")
  print(f"Variance score : \n {metrics['r2']:.2f}")
  print(f"Num shares previsto: {int(examplePrediction)}")

  if showPlots:
    plotRegression3D(model, features, targetArray, show=True)

  return {
    "dataFrame": dataFrame,
    "filtered": filtered,
    "features": features,
    "targets": targetArray,
    "model": model,
    "metrics": metrics,
    "examplePrediction": examplePrediction,
  }


if __name__ == "__main__":
  runPipeline(showPlots=True)

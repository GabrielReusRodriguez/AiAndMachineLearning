"""Clustering K-Means sobre perfiles de personalidades (analisis.csv)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.cluster import KMeans

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_FILE = DATA_DIR / "analisis.csv"
FIRST_N_ROWS = 20
MAX_CLUSTERS_ELBOW = 20
DEFAULT_N_CLUSTERS = 5
FEATURE_COLUMNS = ["op", "ex", "ag"]
CATEGORY_COLUMN = "categoria"
CLUSTER_COLORS = ["red", "green", "blue", "cyan", "yellow"]
CATEGORY_COLORS = [
  "blue",
  "red",
  "green",
  "blue",
  "cyan",
  "yellow",
  "orange",
  "black",
  "pink",
  "brown",
  "purple",
]


def loadAnalisisData(csvPath: Path | str = CSV_FILE) -> pd.DataFrame:
  """Carga el CSV de analisis de personalidades."""
  return pd.read_csv(csvPath)


def summarizeByCategory(dataFrame: pd.DataFrame) -> pd.Series:
  """Cuenta filas por categoria."""
  return dataFrame.groupby(CATEGORY_COLUMN).size()


def extractFeatures(
  dataFrame: pd.DataFrame,
  featureColumns: list[str] | None = None,
) -> np.ndarray:
  """Extrae la matriz de features X a partir de columnas numericas."""
  columns = featureColumns or FEATURE_COLUMNS
  return np.array(dataFrame[columns])


def computeElbowScores(
  features: np.ndarray,
  maxClusters: int = MAX_CLUSTERS_ELBOW,
  randomState: int = 42,
) -> tuple[range, list[float]]:
  """Entrena KMeans para k=1..maxClusters-1 y devuelve (k_range, scores)."""
  if maxClusters < 2:
    raise ValueError("maxClusters debe ser >= 2")
  clusterRange = range(1, maxClusters)
  scores = [
    KMeans(n_clusters=nClusters, n_init=10, random_state=randomState)
    .fit(features)
    .score(features)
    for nClusters in clusterRange
  ]
  return clusterRange, scores


def fitKMeans(
  features: np.ndarray,
  nClusters: int = DEFAULT_N_CLUSTERS,
  randomState: int = 42,
) -> KMeans:
  """Ajusta un modelo KMeans y lo devuelve entrenado."""
  if nClusters < 1:
    raise ValueError("nClusters debe ser >= 1")
  model = KMeans(n_clusters=nClusters, n_init=10, random_state=randomState)
  return model.fit(features)


def predictClusters(model: KMeans, features: np.ndarray) -> np.ndarray:
  """Predice la etiqueta de cluster para cada fila."""
  return model.predict(features)


def plotElbowCurve(
  clusterRange: range,
  scores: list[float],
  show: bool = True,
) -> plt.Figure:
  """Grafica la curva del codo (Elbow Curve)."""
  fig, ax = plt.subplots()
  ax.plot(list(clusterRange), scores)
  ax.set_xlabel("Numero de Clusters")
  ax.set_ylabel("Score")
  ax.set_title("Elbow Curve")
  if show:
    plt.show()
  return fig


def plotFeatureHistograms(dataFrame: pd.DataFrame, show: bool = True):
  """Histograma de columnas numericas (excluye categoria)."""
  axes = dataFrame.drop([CATEGORY_COLUMN], axis="columns").hist(figsize=(10, 8))
  if show:
    plt.show()
  return axes


def plotPairplot(
  dataFrame: pd.DataFrame,
  show: bool = True,
):
  """Pairplot de op/ex/ag coloreado por categoria."""
  grid = sb.pairplot(
    dataFrame.dropna(),
    hue=CATEGORY_COLUMN,
    height=4,
    vars=FEATURE_COLUMNS,
    kind="scatter",
  )
  if show:
    plt.show()
  return grid


def plotFeatures3DByCategory(
  features: np.ndarray,
  categories: np.ndarray,
  show: bool = True,
) -> plt.Figure:
  """Scatter 3D coloreado por categoria original."""
  fig = plt.figure(figsize=(15, 9))
  ax = fig.add_subplot(111, projection="3d")
  colors = [CATEGORY_COLORS[int(row)] for row in categories]
  ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=colors, s=60)
  if show:
    plt.show()
  return fig


def plotClusters3D(
  features: np.ndarray,
  labels: np.ndarray,
  centroids: np.ndarray,
  colors: list[str] | None = None,
  show: bool = True,
) -> plt.Figure:
  """Scatter 3D de puntos y centroides segun cluster asignado."""
  palette = colors or CLUSTER_COLORS
  asignar = [palette[int(row)] for row in labels]
  fig = plt.figure()
  ax = fig.add_subplot(111, projection="3d")
  ax.scatter(features[:, 0], features[:, 1], features[:, 2], c=asignar, s=60)
  ax.scatter(
    centroids[:, 0],
    centroids[:, 1],
    centroids[:, 2],
    marker="*",
    c=palette[: len(centroids)],
    s=1000,
  )
  if show:
    plt.show()
  return fig


def plotClusterPair(
  featureX: np.ndarray,
  featureY: np.ndarray,
  labels: np.ndarray,
  centroidX: np.ndarray,
  centroidY: np.ndarray,
  colors: list[str] | None = None,
  show: bool = True,
) -> plt.Figure:
  """Scatter 2D de un par de features con centroides."""
  palette = colors or CLUSTER_COLORS
  asignar = [palette[int(row)] for row in labels]
  fig, ax = plt.subplots()
  ax.scatter(featureX, featureY, c=asignar, s=70)
  ax.scatter(
    centroidX,
    centroidY,
    marker="*",
    c=palette[: len(centroidX)],
    s=1000,
  )
  if show:
    plt.show()
  return fig


def runPipeline(showPlots: bool = True) -> dict:
  """Ejecuta carga, exploracion, elbow y clustering K-Means."""
  dataFrame = loadAnalisisData()
  print(dataFrame.describe())
  print(dataFrame.head(FIRST_N_ROWS))
  print(summarizeByCategory(dataFrame))

  if showPlots:
    plotFeatureHistograms(dataFrame, show=True)
    plotPairplot(dataFrame, show=True)

  features = extractFeatures(dataFrame)
  categories = np.array(dataFrame[CATEGORY_COLUMN])

  if showPlots:
    plotFeatures3DByCategory(features, categories, show=True)

  clusterRange, scores = computeElbowScores(features)
  if showPlots:
    plotElbowCurve(clusterRange, scores, show=True)

  model = fitKMeans(features, nClusters=DEFAULT_N_CLUSTERS)
  labels = predictClusters(model, features)
  centroids = model.cluster_centers_
  print(f"Centroides: {centroids}", end="\n\n")
  print(f"Labels: {labels}", end="\n\n")

  if showPlots:
    plotClusters3D(features, labels, centroids, show=True)
    plotClusterPair(
      dataFrame["op"].values,
      dataFrame["ex"].values,
      labels,
      centroids[:, 0],
      centroids[:, 1],
      show=True,
    )
    plotClusterPair(
      dataFrame["op"].values,
      dataFrame["ag"].values,
      labels,
      centroids[:, 0],
      centroids[:, 2],
      show=True,
    )
    plotClusterPair(
      dataFrame["ex"].values,
      dataFrame["ag"].values,
      labels,
      centroids[:, 1],
      centroids[:, 2],
      show=True,
    )

  return {
    "dataFrame": dataFrame,
    "features": features,
    "model": model,
    "labels": labels,
    "centroids": centroids,
    "elbowScores": scores,
  }


if __name__ == "__main__":
  runPipeline(showPlots=True)

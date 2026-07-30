"""Filtrado colaborativo k-NN para recomendar repositorios GitHub."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_RATINGS_FILE = DATA_DIR / "ratings.csv"
CSV_REPOS_FILE = DATA_DIR / "repos.csv"
CSV_USERS_FILE = DATA_DIR / "users.csv"
N_FIRST_ROWS = 10
DEFAULT_N_NEIGHBORS = 5
DEFAULT_TOP_N = 10
DEFAULT_TEST_SIZE = 0.2
USER_COLUMN = "userId"
REPO_COLUMN = "repoId"
RATING_COLUMN = "rating"


def loadRatings(csvPath: Path | str = CSV_RATINGS_FILE) -> pd.DataFrame:
  """Carga el CSV de valoraciones usuario-repositorio."""
  return pd.read_csv(csvPath)


def loadRepos(csvPath: Path | str = CSV_REPOS_FILE) -> pd.DataFrame:
  """Carga el catálogo de repositorios."""
  return pd.read_csv(csvPath)


def loadUsers(csvPath: Path | str = CSV_USERS_FILE) -> pd.DataFrame:
  """Carga el catálogo de usuarios."""
  return pd.read_csv(csvPath)


def countUniqueUsersAndRepos(dfRatings: pd.DataFrame) -> tuple[int, int]:
  """Devuelve (usuarios únicos, repos únicos con rating)."""
  nUsers = dfRatings[USER_COLUMN].nunique()
  nRepos = dfRatings[REPO_COLUMN].nunique()
  return nUsers, nRepos


def buildUserItemMatrix(dfRatings: pd.DataFrame) -> pd.DataFrame:
  """Construye la matriz usuario-ítem (0 donde no hay valoración)."""
  return dfRatings.pivot_table(
    index=USER_COLUMN,
    columns=REPO_COLUMN,
    values=RATING_COLUMN,
    fill_value=0,
  )


def fitUserNeighbors(
  userItemMatrix: pd.DataFrame,
  nNeighbors: int = DEFAULT_N_NEIGHBORS,
) -> NearestNeighbors:
  """Ajusta k-NN sobre vectores de valoraciones por usuario."""
  if nNeighbors < 1:
    raise ValueError("nNeighbors debe ser >= 1")
  nSamples = len(userItemMatrix)
  effectiveNeighbors = min(nNeighbors + 1, nSamples)
  model = NearestNeighbors(metric="cosine", n_neighbors=effectiveNeighbors)
  model.fit(userItemMatrix.values)
  return model


def findSimilarUsers(
  userItemMatrix: pd.DataFrame,
  neighborsModel: NearestNeighbors,
  userId: int,
  nNeighbors: int = DEFAULT_N_NEIGHBORS,
) -> list[int]:
  """Devuelve ids de usuarios similares (excluye al propio usuario)."""
  if userId not in userItemMatrix.index:
    raise ValueError(f"userId {userId} no encontrado")
  userIdx = userItemMatrix.index.get_loc(userId)
  _, indices = neighborsModel.kneighbors(userItemMatrix.iloc[[userIdx]].values)
  neighborIds = [
    int(userItemMatrix.index[i])
    for i in indices[0]
    if i != userIdx
  ]
  return neighborIds[:nNeighbors]


def predictRating(
  userItemMatrix: pd.DataFrame,
  neighborsModel: NearestNeighbors,
  userId: int,
  repoId: int,
  nNeighbors: int = DEFAULT_N_NEIGHBORS,
) -> float | None:
  """Predice la valoración de un usuario sobre un repo vía vecinos."""
  if userId not in userItemMatrix.index or repoId not in userItemMatrix.columns:
    return None
  userIdx = userItemMatrix.index.get_loc(userId)
  _, indices = neighborsModel.kneighbors(userItemMatrix.iloc[[userIdx]].values)
  neighborIndices = [i for i in indices[0] if i != userIdx][:nNeighbors]
  ratings = [
    userItemMatrix.iloc[ni][repoId]
    for ni in neighborIndices
    if userItemMatrix.iloc[ni][repoId] > 0
  ]
  if not ratings:
    return None
  return float(np.mean(ratings))


def recommendRepos(
  userItemMatrix: pd.DataFrame,
  neighborsModel: NearestNeighbors,
  userId: int,
  nNeighbors: int = DEFAULT_N_NEIGHBORS,
  topN: int = DEFAULT_TOP_N,
) -> list[tuple[int, float]]:
  """Recomienda repos no valorados, ordenados por rating predicho."""
  if userId not in userItemMatrix.index:
    raise ValueError(f"userId {userId} no encontrado")
  userRatings = userItemMatrix.loc[userId]
  unratedRepos = userRatings[userRatings == 0].index
  predictions: list[tuple[int, float]] = []
  for repoId in unratedRepos:
    predicted = predictRating(
      userItemMatrix,
      neighborsModel,
      userId,
      int(repoId),
      nNeighbors=nNeighbors,
    )
    if predicted is not None:
      predictions.append((int(repoId), predicted))
  predictions.sort(key=lambda item: item[1], reverse=True)
  return predictions[:topN]


def evaluateMse(
  dfRatings: pd.DataFrame,
  testSize: float = DEFAULT_TEST_SIZE,
  nNeighbors: int = DEFAULT_N_NEIGHBORS,
  randomState: int = 42,
) -> float:
  """Evalúa MSE en hold-out sobre filas de ratings."""
  if not 0 < testSize < 1:
    raise ValueError("testSize debe estar entre 0 y 1")
  train, test = train_test_split(
    dfRatings,
    test_size=testSize,
    random_state=randomState,
  )
  userItemMatrix = buildUserItemMatrix(train)
  neighborsModel = fitUserNeighbors(userItemMatrix, nNeighbors)
  predictions: list[float] = []
  actuals: list[float] = []
  for _, row in test.iterrows():
    actual = float(row[RATING_COLUMN])
    predicted = predictRating(
      userItemMatrix,
      neighborsModel,
      int(row[USER_COLUMN]),
      int(row[REPO_COLUMN]),
      nNeighbors=nNeighbors,
    )
    if predicted is not None:
      predictions.append(predicted)
      actuals.append(actual)
  if not actuals:
    raise ValueError("No hay predicciones válidas para calcular MSE")
  return float(mean_squared_error(actuals, predictions))


def plotRatingHistogram(
  dfRatings: pd.DataFrame,
  bins: int = 8,
  show: bool = True,
) -> plt.Figure:
  """Histograma de distribución de puntuaciones."""
  fig, ax = plt.subplots()
  ax.hist(dfRatings[RATING_COLUMN], bins=bins)
  ax.set_xlabel("Rating")
  ax.set_ylabel("Frecuencia")
  ax.set_title("Distribución de ratings")
  if show:
    plt.show()
  return fig


def runPipeline(
  userId: int = 1,
  nNeighbors: int = DEFAULT_N_NEIGHBORS,
  topN: int = DEFAULT_TOP_N,
  showPlots: bool = True,
) -> dict:
  """Ejecuta carga, exploración, k-NN y recomendaciones."""
  dfRatings = loadRatings()
  dfRepos = loadRepos()
  dfUsers = loadUsers()

  print(dfRatings.describe())
  print(dfRatings.head(N_FIRST_ROWS))
  print(dfRepos.describe())
  print(dfRepos.head(N_FIRST_ROWS))
  print(dfUsers.describe())
  print(dfUsers.head(N_FIRST_ROWS))

  nUsers, nRepos = countUniqueUsersAndRepos(dfRatings)
  print(f"Número de usuarios únicos: {nUsers}")
  print(f"Número de repos valorados: {nRepos}")

  if showPlots:
    plotRatingHistogram(dfRatings, show=True)

  userItemMatrix = buildUserItemMatrix(dfRatings)
  neighborsModel = fitUserNeighbors(userItemMatrix, nNeighbors)
  similarUsers = findSimilarUsers(
    userItemMatrix,
    neighborsModel,
    userId,
    nNeighbors=nNeighbors,
  )
  recommendations = recommendRepos(
    userItemMatrix,
    neighborsModel,
    userId,
    nNeighbors=nNeighbors,
    topN=topN,
  )
  mse = evaluateMse(dfRatings, nNeighbors=nNeighbors)

  print(f"Usuarios similares a {userId}: {similarUsers}")
  print(f"Recomendaciones para usuario {userId}: {recommendations}")
  print(f"MSE (hold-out): {mse:.4f}")

  return {
    "dfRatings": dfRatings,
    "dfRepos": dfRepos,
    "dfUsers": dfUsers,
    "userItemMatrix": userItemMatrix,
    "neighborsModel": neighborsModel,
    "similarUsers": similarUsers,
    "recommendations": recommendations,
    "mse": mse,
  }


if __name__ == "__main__":
  runPipeline(showPlots=True)

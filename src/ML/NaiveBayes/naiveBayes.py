"""Clasificador Gaussian Naive Bayes: comprar vs alquilar vivienda."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_FILE = DATA_DIR / "comprar_alquilar.csv"
N_FIRST_ROWS = 10
N_BEST_COLUMNS = 5
TARGET_COLUMN = "comprar"
TEST_SIZE = 0.2
RANDOM_STATE = 6
EXPECTED_FEATURE_COLUMNS = [
  "ingresos",
  "gastos_comunes",
  "pago_coche",
  "gastos_otros",
  "ahorros",
  "vivienda",
  "estado_civil",
  "hijos",
  "trabajo",
]


def loadComprarAlquilarData(csvPath: Path | str = CSV_FILE) -> pd.DataFrame:
  """Carga el CSV de perfiles financieros etiquetados."""
  return pd.read_csv(csvPath)


def addDerivedColumns(dataFrame: pd.DataFrame) -> pd.DataFrame:
  """Añade columnas derivadas gastos e inversion (copia del DataFrame)."""
  result = dataFrame.copy()
  result["gastos"] = (
    result["gastos_comunes"] + result["gastos_otros"] + result["pago_coche"]
  )
  result["inversion"] = result["vivienda"] - result["ahorros"]
  return result


def splitFeaturesTarget(
  dataFrame: pd.DataFrame,
  targetColumn: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.Series]:
  """Separa features X y target y."""
  features = dataFrame.drop([targetColumn], axis="columns")
  target = dataFrame[targetColumn]
  return features, target


def selectBestFeatures(
  features: pd.DataFrame,
  target: pd.Series,
  nBest: int = N_BEST_COLUMNS,
) -> tuple[list[str], SelectKBest, np.ndarray]:
  """Selecciona las nBest features con SelectKBest y devuelve nombres y matriz."""
  if nBest < 1:
    raise ValueError("nBest debe ser >= 1")
  if nBest > features.shape[1]:
    raise ValueError("nBest no puede superar el numero de columnas")
  selector = SelectKBest(k=nBest)
  transformed = selector.fit_transform(features, target)
  selectedIndices = selector.get_support(indices=True)
  selectedColumns = list(features.columns[selectedIndices])
  return selectedColumns, selector, transformed


def splitTrainTest(
  dataFrame: pd.DataFrame,
  testSize: float = TEST_SIZE,
  randomState: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Divide el dataset en conjuntos de entrenamiento y prueba."""
  if not 0 < testSize < 1:
    raise ValueError("testSize debe estar entre 0 y 1")
  return train_test_split(
    dataFrame,
    test_size=testSize,
    random_state=randomState,
  )


def fitGaussianNB(
  features: np.ndarray,
  target: pd.Series | np.ndarray,
) -> GaussianNB:
  """Entrena un clasificador GaussianNB."""
  model = GaussianNB()
  return model.fit(features, target)


def predictComprar(model: GaussianNB, features: np.ndarray) -> np.ndarray:
  """Predice la etiqueta comprar (1) o alquilar (0)."""
  return model.predict(features)


def scoreModel(
  model: GaussianNB,
  features: np.ndarray,
  target: pd.Series | np.ndarray,
) -> float:
  """Devuelve la precision (accuracy) del modelo sobre X, y."""
  return float(model.score(features, target))


def plotFeatureHistograms(
  dataFrame: pd.DataFrame,
  targetColumn: str = TARGET_COLUMN,
  show: bool = True,
):
  """Histogramas de features excluyendo la columna target."""
  axes = dataFrame.drop([targetColumn], axis="columns").hist(figsize=(20, 10))
  if show:
    plt.show()
  return axes


def plotFeatureCorrelation(
  dataFrame: pd.DataFrame,
  featureColumns: list[str],
  title: str = "Correlacion de Pearson entre features",
  show: bool = True,
) -> plt.Figure:
  """Mapa de calor de correlacion entre columnas seleccionadas."""
  colormap = plt.cm.viridis
  fig, _ = plt.subplots(figsize=(12, 12))
  plt.title(title, y=1.05, size=15)
  sb.heatmap(
    dataFrame[featureColumns].astype(float).corr(),
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=True,
  )
  if show:
    plt.show()
  return fig


def runPipeline(showPlots: bool = False) -> dict:
  """Ejecuta carga, feature selection, entrenamiento y evaluacion."""
  dataFrame = addDerivedColumns(loadComprarAlquilarData())
  print(dataFrame.describe())
  print(dataFrame.head(N_FIRST_ROWS))

  if showPlots:
    plotFeatureHistograms(dataFrame, show=True)

  features, target = splitFeaturesTarget(dataFrame)
  selectedColumns, _, _ = selectBestFeatures(features, target)

  if showPlots:
    plotFeatureCorrelation(dataFrame, selectedColumns, show=True)
    allFeatures = list(features.columns)
    plotFeatureCorrelation(
      dataFrame,
      allFeatures,
      title="Correlacion de Pearson entre todas las features",
      show=True,
    )

  trainFrame, testFrame = splitTrainTest(dataFrame)
  trainFeatures = trainFrame[selectedColumns].values
  testFeatures = testFrame[selectedColumns].values
  trainTarget = trainFrame[TARGET_COLUMN]
  testTarget = testFrame[TARGET_COLUMN]

  model = fitGaussianNB(trainFeatures, trainTarget)
  predictions = predictComprar(model, testFeatures)

  trainScore = scoreModel(model, trainFeatures, trainTarget)
  testScore = scoreModel(model, testFeatures, testTarget)
  print(f"Precision del set de Entrenamiento: {trainScore:.2f}")
  print(f"Precision del set de Test: {testScore:.2f}")
  print(classification_report(testTarget, predictions))
  print(confusion_matrix(testTarget, predictions))

  return {
    "dataFrame": dataFrame,
    "selectedColumns": selectedColumns,
    "model": model,
    "trainScore": trainScore,
    "testScore": testScore,
    "predictions": predictions,
    "trainFrame": trainFrame,
    "testFrame": testFrame,
  }


if __name__ == "__main__":
  runPipeline(showPlots=True)

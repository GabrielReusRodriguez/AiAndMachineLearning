"""Clasificación multiclase con regresión logística (usuarios web por SO)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_DIR = Path(__file__).resolve().parent / "data"
CSV_FILE = DATA_DIR / "usuarios_win_mac_lin.csv"
FEATURE_COLUMNS = ["duracion", "paginas", "acciones", "valor"]
LABEL_COLUMN = "clase"
VALIDATION_SIZE = 0.2
RANDOM_STATE = 7
CV_FOLDS = 10


def loadUsuariosData(csvPath: Path | str = CSV_FILE) -> pd.DataFrame:
  """Carga el CSV de comportamiento web etiquetado por sistema operativo."""
  return pd.read_csv(csvPath)


def summarizeByClass(dataFrame: pd.DataFrame) -> pd.Series:
  """Cuenta filas por clase (0 Windows, 1 Mac, 2 Linux)."""
  return dataFrame.groupby(LABEL_COLUMN).size()


def extractFeaturesAndLabels(
  dataFrame: pd.DataFrame,
  featureColumns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
  """Extrae la matriz de features X y el vector de etiquetas y."""
  columns = featureColumns or FEATURE_COLUMNS
  features = np.array(dataFrame[columns])
  labels = np.array(dataFrame[LABEL_COLUMN])
  return features, labels


def fitLogisticModel(
  features: np.ndarray,
  labels: np.ndarray,
) -> LogisticRegression:
  """Ajusta un modelo LogisticRegression multiclase."""
  model = LogisticRegression(max_iter=1000)
  return model.fit(features, labels)


def splitTrainValidation(
  features: np.ndarray,
  labels: np.ndarray,
  testSize: float = VALIDATION_SIZE,
  randomState: int = RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Divide en entrenamiento y validación (hold-out)."""
  return model_selection.train_test_split(
    features,
    labels,
    test_size=testSize,
    random_state=randomState,
  )


def crossValidateAccuracy(
  model: LogisticRegression,
  features: np.ndarray,
  labels: np.ndarray,
  nSplits: int = CV_FOLDS,
) -> np.ndarray:
  """Validación cruzada K-Fold; devuelve scores de accuracy por fold."""
  kfold = model_selection.KFold(n_splits=nSplits)
  return model_selection.cross_val_score(
    model,
    features,
    labels,
    cv=kfold,
    scoring="accuracy",
  )


def evaluateHoldOut(
  model: LogisticRegression,
  features: np.ndarray,
  labels: np.ndarray,
) -> dict:
  """Predice sobre hold-out y devuelve métricas de clasificación."""
  predictions = model.predict(features)
  return {
    "predictions": predictions,
    "accuracy": accuracy_score(labels, predictions),
    "confusionMatrix": confusion_matrix(labels, predictions),
    "classificationReport": classification_report(labels, predictions),
  }


def plotFeatureHistograms(dataFrame: pd.DataFrame, show: bool = True):
  """Histograma de columnas numéricas (excluye clase)."""
  axes = dataFrame.drop([LABEL_COLUMN], axis="columns").hist(figsize=(10, 8))
  if show:
    plt.show()
  return axes


def plotPairplot(dataFrame: pd.DataFrame, show: bool = True):
  """Pairplot de features coloreado por clase."""
  grid = sb.pairplot(
    dataFrame.dropna(),
    hue=LABEL_COLUMN,
    height=4,
    vars=FEATURE_COLUMNS,
    kind="reg",
  )
  if show:
    plt.show()
  return grid


def runPipeline(showPlots: bool = True) -> dict:
  """Ejecuta exploración, entrenamiento y validación del modelo multiclase."""
  dataFrame = loadUsuariosData()
  print(dataFrame.head())
  print(dataFrame.describe())
  print(summarizeByClass(dataFrame))

  if showPlots:
    plotFeatureHistograms(dataFrame, show=True)
    plotPairplot(dataFrame, show=True)

  features, labels = extractFeaturesAndLabels(dataFrame)
  model = fitLogisticModel(features, labels)
  print(f"model Score: {model.score(features, labels)}")

  X_train, X_validation, y_train, y_validation = splitTrainValidation(
    features,
    labels,
  )
  cvScores = crossValidateAccuracy(model, X_train, y_train)
  print(
    "Logistic Regression: %f (%f)" % (cvScores.mean(), cvScores.std())
  )

  metrics = evaluateHoldOut(model, X_validation, y_validation)
  print(metrics["accuracy"])
  print(metrics["confusionMatrix"])
  print(metrics["classificationReport"])

  return {
    "dataFrame": dataFrame,
    "features": features,
    "labels": labels,
    "model": model,
    "cvScores": cvScores,
    "metrics": metrics,
  }


if __name__ == "__main__":
  runPipeline(showPlots=True)

"""Regresión logística binaria con datos sintéticos y frontera de decisión."""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

DEFAULT_NUM_SAMPLES = 100
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def generateSyntheticData(
  numSamples: int = DEFAULT_NUM_SAMPLES,
  randomState: int = DEFAULT_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray]:
  """Genera features 2D y etiquetas binarias (suma de features > 10)."""
  rng = np.random.default_rng(randomState)
  features = rng.random((numSamples, 2)) * 10
  labels = (features[:, 0] + features[:, 1] > 10).astype(int)
  return features, labels


def splitSyntheticData(
  features: np.ndarray,
  labels: np.ndarray,
  testSize: float = DEFAULT_TEST_SIZE,
  randomState: int = DEFAULT_RANDOM_STATE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Divide datos sintéticos en entrenamiento y prueba."""
  return train_test_split(
    features,
    labels,
    test_size=testSize,
    random_state=randomState,
  )


def fitBinaryLogisticModel(
  features: np.ndarray,
  labels: np.ndarray,
) -> LogisticRegression:
  """Ajusta LogisticRegression para clasificación binaria."""
  model = LogisticRegression()
  return model.fit(features, labels)


def evaluateBinaryModel(
  model: LogisticRegression,
  features: np.ndarray,
  labels: np.ndarray,
) -> dict:
  """Evalúa el modelo binario y devuelve accuracy, matriz y reporte."""
  predictions = model.predict(features)
  return {
    "predictions": predictions,
    "accuracy": accuracy_score(labels, predictions),
    "confusionMatrix": confusion_matrix(labels, predictions),
    "classificationReport": classification_report(labels, predictions),
  }


def plotDecisionBoundary(
  model: LogisticRegression,
  features: np.ndarray,
  labels: np.ndarray,
  show: bool = True,
  gridStep: float = 0.1,
) -> plt.Figure:
  """Dibuja la frontera de decisión 2D del clasificador."""
  xMin, xMax = features[:, 0].min() - 1, features[:, 0].max() + 1
  yMin, yMax = features[:, 1].min() - 1, features[:, 1].max() + 1
  xx, yy = np.meshgrid(
    np.arange(xMin, xMax, gridStep),
    np.arange(yMin, yMax, gridStep),
  )
  gridPredictions = model.predict(np.c_[xx.ravel(), yy.ravel()])
  gridPredictions = gridPredictions.reshape(xx.shape)

  fig, ax = plt.subplots()
  ax.contourf(xx, yy, gridPredictions, alpha=0.8, cmap=plt.cm.Paired)
  ax.scatter(
    features[:, 0],
    features[:, 1],
    c=labels,
    edgecolors="k",
    cmap=plt.cm.Paired,
  )
  ax.set_title("Logistic Regression Decision Boundary")
  ax.set_xlabel("Feature 1")
  ax.set_ylabel("Feature 2")
  if show:
    plt.show()
  return fig


def runSyntheticPipeline(
  showPlots: bool = True,
  numSamples: int = DEFAULT_NUM_SAMPLES,
  randomState: int = DEFAULT_RANDOM_STATE,
) -> dict:
  """Pipeline completo: datos sintéticos, entrenamiento, evaluación y plot."""
  features, labels = generateSyntheticData(numSamples, randomState)
  X_train, X_test, y_train, y_test = splitSyntheticData(
    features,
    labels,
    randomState=randomState,
  )

  model = fitBinaryLogisticModel(X_train, y_train)
  metrics = evaluateBinaryModel(model, X_test, y_test)

  print("Accuracy:", metrics["accuracy"])
  print("Confusion Matrix:\n", metrics["confusionMatrix"])
  print("Classification Report:\n", metrics["classificationReport"])

  if showPlots:
    plotDecisionBoundary(model, features, labels, show=True)

  return {
    "features": features,
    "labels": labels,
    "model": model,
    "metrics": metrics,
  }


if __name__ == "__main__":
  runSyntheticPipeline(showPlots=True)

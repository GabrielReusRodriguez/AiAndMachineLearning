"""Tests para src/ML/RegresionLogistica."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / "src" / "ML" / "RegresionLogistica"
sys.path.insert(0, str(MODULE_DIR))

from regresionLogistica import (  # noqa: E402
  CSV_FILE,
  FEATURE_COLUMNS,
  LABEL_COLUMN,
  crossValidateAccuracy,
  evaluateHoldOut,
  extractFeaturesAndLabels,
  fitLogisticModel,
  loadUsuariosData,
  splitTrainValidation,
  summarizeByClass,
)
from regresionLogistica_v2 import (  # noqa: E402
  DEFAULT_RANDOM_STATE,
  evaluateBinaryModel,
  fitBinaryLogisticModel,
  generateSyntheticData,
  runSyntheticPipeline,
  splitSyntheticData,
)


def testLoadUsuariosDataHasExpectedColumns():
  dataFrame = loadUsuariosData(CSV_FILE)
  expected = set(FEATURE_COLUMNS) | {LABEL_COLUMN}
  assert expected.issubset(set(dataFrame.columns))
  assert len(dataFrame) > 0


def testSummarizeByClassCountsMatch():
  dataFrame = pd.DataFrame(
    {
      "duracion": [1, 2, 3],
      "paginas": [1, 1, 1],
      "acciones": [1, 1, 1],
      "valor": [1, 1, 1],
      "clase": [0, 0, 2],
    }
  )
  counts = summarizeByClass(dataFrame)
  assert counts.loc[0] == 2
  assert counts.loc[2] == 1


def testExtractFeaturesAndLabelsShapeAndValues():
  dataFrame = pd.DataFrame(
    {
      "duracion": [7.0, 21.0],
      "paginas": [2.0, 2.0],
      "acciones": [4.0, 6.0],
      "valor": [8.0, 6.0],
      "clase": [2, 2],
    }
  )
  features, labels = extractFeaturesAndLabels(dataFrame)
  assert features.shape == (2, 4)
  assert np.allclose(features[0], [7.0, 2.0, 4.0, 8.0])
  assert list(labels) == [2, 2]
  assert list(FEATURE_COLUMNS) == ["duracion", "paginas", "acciones", "valor"]


def testFitLogisticModelPredictsThreeClasses():
  features = np.array(
    [
      [1.0, 1.0, 1.0, 1.0],
      [2.0, 2.0, 2.0, 2.0],
      [10.0, 10.0, 10.0, 10.0],
      [11.0, 11.0, 11.0, 11.0],
      [20.0, 20.0, 20.0, 20.0],
      [21.0, 21.0, 21.0, 21.0],
    ]
  )
  labels = np.array([0, 0, 1, 1, 2, 2])
  model = fitLogisticModel(features, labels)
  predictions = model.predict(features)
  assert predictions.shape == (6,)
  assert set(predictions) <= {0, 1, 2}


def testCrossValidateAccuracyReturnsExpectedLength():
  features, labels = extractFeaturesAndLabels(loadUsuariosData(CSV_FILE))
  model = fitLogisticModel(features, labels)
  X_train, _, y_train, _ = splitTrainValidation(features, labels)
  cvScores = crossValidateAccuracy(model, X_train, y_train, nSplits=5)
  assert len(cvScores) == 5
  assert np.all((cvScores >= 0) & (cvScores <= 1))


def testEvaluateHoldOutReturnsMetrics():
  features = np.array([[1.0, 1.0, 1.0, 1.0], [10.0, 10.0, 10.0, 10.0]])
  labels = np.array([0, 1])
  model = fitLogisticModel(features, labels)
  metrics = evaluateHoldOut(model, features, labels)
  assert metrics["accuracy"] == 1.0
  assert metrics["confusionMatrix"].shape == (2, 2)
  assert "precision" in metrics["classificationReport"]


def testPipelineConstants():
  assert CSV_FILE.exists()
  assert LABEL_COLUMN == "clase"


def testGenerateSyntheticDataShapeAndLabels():
  features, labels = generateSyntheticData(numSamples=50, randomState=0)
  assert features.shape == (50, 2)
  assert labels.shape == (50,)
  assert set(labels).issubset({0, 1})


def testSplitSyntheticDataPreservesTotalRows():
  features, labels = generateSyntheticData(numSamples=100, randomState=0)
  X_train, X_test, y_train, y_test = splitSyntheticData(
    features,
    labels,
    testSize=0.2,
    randomState=DEFAULT_RANDOM_STATE,
  )
  assert len(X_train) + len(X_test) == 100
  assert len(y_train) + len(y_test) == 100


def testFitBinaryLogisticModelAndEvaluate():
  features, labels = generateSyntheticData(numSamples=200, randomState=42)
  X_train, X_test, y_train, y_test = splitSyntheticData(features, labels)
  model = fitBinaryLogisticModel(X_train, y_train)
  metrics = evaluateBinaryModel(model, X_test, y_test)
  assert metrics["accuracy"] >= 0.5
  assert metrics["confusionMatrix"].shape == (2, 2)


def testRunSyntheticPipelineWithoutPlots():
  result = runSyntheticPipeline(showPlots=False, numSamples=80, randomState=7)
  assert "model" in result
  assert "metrics" in result
  assert result["features"].shape[0] == 80

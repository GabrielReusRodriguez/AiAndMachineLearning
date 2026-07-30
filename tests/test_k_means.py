"""Tests para src/ML/K-means."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / "src" / "ML" / "K-means"
sys.path.insert(0, str(MODULE_DIR))

from kMeans import (  # noqa: E402
  CSV_FILE,
  DEFAULT_N_CLUSTERS,
  FEATURE_COLUMNS,
  computeElbowScores,
  extractFeatures,
  fitKMeans,
  loadAnalisisData,
  predictClusters,
  summarizeByCategory,
)


def testLoadAnalisisDataHasExpectedColumns():
  dataFrame = loadAnalisisData(CSV_FILE)
  expected = {"usuario", "op", "co", "ex", "ag", "ne", "wordcount", "categoria"}
  assert expected.issubset(set(dataFrame.columns))
  assert len(dataFrame) > 0


def testSummarizeByCategoryCountsMatch():
  dataFrame = pd.DataFrame(
    {
      "categoria": [1, 1, 2, 7, 7, 7],
      "op": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    }
  )
  counts = summarizeByCategory(dataFrame)
  assert counts.loc[1] == 2
  assert counts.loc[2] == 1
  assert counts.loc[7] == 3


def testExtractFeaturesShapeAndValues():
  dataFrame = pd.DataFrame(
    {
      "op": [1.0, 2.0],
      "ex": [3.0, 4.0],
      "ag": [5.0, 6.0],
      "categoria": [1, 2],
    }
  )
  features = extractFeatures(dataFrame)
  assert features.shape == (2, 3)
  assert np.allclose(features[0], [1.0, 3.0, 5.0])
  assert list(FEATURE_COLUMNS) == ["op", "ex", "ag"]


def testComputeElbowScoresLengthAndMonotonicTrend():
  rng = np.random.default_rng(0)
  features = rng.normal(size=(80, 3))
  clusterRange, scores = computeElbowScores(features, maxClusters=6)
  assert list(clusterRange) == [1, 2, 3, 4, 5]
  assert len(scores) == 5
  # El score de KMeans (negativo de inercia) no empeora al subir k
  assert scores[-1] >= scores[0]


def testComputeElbowScoresRejectsInvalidMaxClusters():
  features = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  with pytest.raises(ValueError, match="maxClusters"):
    computeElbowScores(features, maxClusters=1)


def testFitKMeansAndPredictClusters():
  features = np.array(
    [
      [0.0, 0.0, 0.0],
      [0.1, 0.1, 0.1],
      [10.0, 10.0, 10.0],
      [10.1, 10.1, 10.1],
    ]
  )
  model = fitKMeans(features, nClusters=2, randomState=0)
  labels = predictClusters(model, features)
  assert model.cluster_centers_.shape == (2, 3)
  assert labels.shape == (4,)
  assert set(labels) == {0, 1}
  # Puntos cercanos deben compartir cluster
  assert labels[0] == labels[1]
  assert labels[2] == labels[3]
  assert labels[0] != labels[2]


def testFitKMeansRejectsInvalidNClusters():
  features = np.array([[1.0, 2.0, 3.0]])
  with pytest.raises(ValueError, match="nClusters"):
    fitKMeans(features, nClusters=0)


def testPipelineConstants():
  assert DEFAULT_N_CLUSTERS == 5
  assert CSV_FILE.exists()

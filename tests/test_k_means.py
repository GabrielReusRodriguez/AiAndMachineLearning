"""Tests para src/ML/K-means."""

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

MODULE_DIR = Path(__file__).resolve().parents[1] / "src" / "ML" / "K-means"
sys.path.insert(0, str(MODULE_DIR))

from kMeans import (  # noqa: E402
  CSV_FILE,
  DEFAULT_N_CLUSTERS,
  FEATURE_COLUMNS,
  MAX_CLUSTERS_ELBOW,
  RANDOM_STATE,
  computeElbowScores,
  extractFeatures,
  fitKMeans,
  loadAnalisisData,
  plotClusterPair,
  plotClusters3D,
  plotElbowCurve,
  plotFeatureHistograms,
  plotFeatures3DByCategory,
  plotPairplot,
  predictClusters,
  runPipeline,
  summarizeByCategory,
)


def testLoadAnalisisDataHasExpectedColumns():
  dataFrame = loadAnalisisData(CSV_FILE)
  expected = {"usuario", "op", "co", "ex", "ag", "ne", "wordcount", "categoria"}
  assert expected.issubset(set(dataFrame.columns))
  assert len(dataFrame) == 140


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


def testRunPipelineOnRealDataWithoutPlots():
  result = runPipeline(showPlots=False)
  assert len(result["dataFrame"]) == 140
  assert result["features"].shape == (140, 3)
  assert result["labels"].shape == (140,)
  assert result["centroids"].shape == (DEFAULT_N_CLUSTERS, 3)
  assert len(set(result["labels"])) == DEFAULT_N_CLUSTERS
  assert len(result["elbowScores"]) == MAX_CLUSTERS_ELBOW - 1


def testPlotFunctionsReturnWithoutShow():
  dataFrame = loadAnalisisData()
  features = extractFeatures(dataFrame)
  categories = np.array(dataFrame["categoria"])
  clusterRange, scores = computeElbowScores(features, maxClusters=4)
  model = fitKMeans(features, nClusters=2, randomState=RANDOM_STATE)
  labels = predictClusters(model, features)
  centroids = model.cluster_centers_

  assert plotFeatureHistograms(dataFrame, show=False) is not None
  assert plotPairplot(dataFrame, show=False) is not None
  assert plotFeatures3DByCategory(features, categories, show=False) is not None
  assert plotElbowCurve(clusterRange, scores, show=False) is not None
  assert plotClusters3D(features, labels, centroids, show=False) is not None
  assert plotClusterPair(
    features[:, 0],
    features[:, 1],
    labels,
    centroids[:, 0],
    centroids[:, 1],
    show=False,
  ) is not None


def testPipelineConstants():
  assert DEFAULT_N_CLUSTERS == 5
  assert MAX_CLUSTERS_ELBOW == 20
  assert RANDOM_STATE == 42
  assert CSV_FILE.exists()

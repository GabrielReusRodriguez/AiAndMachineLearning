"""Tests para src/ML/LinearRegression2var."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / "src" / "ML" / "LinearRegression2var"
sys.path.insert(0, str(MODULE_DIR))

from linealRegression2var import (  # noqa: E402
  CSV_FILE,
  EXAMPLE_ENGAGEMENT,
  EXAMPLE_WORD_COUNT,
  FEATURE_COLUMNS,
  MAX_SHARES,
  MAX_WORD_COUNT,
  TARGET_COLUMN,
  buildEngagementSum,
  buildFeatureFrame,
  evaluateRegression,
  extractTrainingArrays,
  filterArticles,
  fitLinearRegression,
  loadArticlesData,
  predictShares,
  runPipeline,
)


def testLoadArticlesDataHasExpectedColumns():
  dataFrame = loadArticlesData(CSV_FILE)
  expected = {
    "Title",
    "url",
    "Word count",
    "# of Links",
    "# of comments",
    "# Images video",
    "Elapsed days",
    TARGET_COLUMN,
  }
  assert expected.issubset(set(dataFrame.columns))
  assert len(dataFrame) > 0


def testFilterArticlesRemovesOutliers():
  dataFrame = pd.DataFrame(
    {
      "Word count": [100, 4000, 500],
      TARGET_COLUMN: [100, 200, 90000],
    }
  )
  filtered = filterArticles(dataFrame)
  assert len(filtered) == 1
  assert filtered.iloc[0]["Word count"] == 100


def testBuildEngagementSumHandlesNaN():
  dataFrame = pd.DataFrame(
    {
      "# of Links": [1, 2],
      "# of comments": [np.nan, 3],
      "# Images video": [4, 5],
    }
  )
  sums = buildEngagementSum(dataFrame)
  assert sums.iloc[0] == 5
  assert sums.iloc[1] == 10


def testBuildFeatureFrameColumns():
  filtered = pd.DataFrame(
    {
      "Word count": [100, 200],
      "# of Links": [1, 2],
      "# of comments": [0, 1],
      "# Images video": [2, 3],
    }
  )
  features = buildFeatureFrame(filtered)
  assert list(features.columns) == FEATURE_COLUMNS
  assert features.loc[0, "suma"] == 3
  assert features.loc[1, "suma"] == 6


def testExtractTrainingArraysShape():
  featureFrame = pd.DataFrame({"Word count": [1, 2], "suma": [3, 4]})
  targets = pd.Series([10, 20])
  features, targetArray = extractTrainingArrays(featureFrame, targets)
  assert features.shape == (2, 2)
  assert targetArray.shape == (2,)


def testFitAndEvaluateOnSyntheticData():
  features = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
  targets = np.array([5.0, 9.0, 13.0])
  model = fitLinearRegression(features, targets)
  metrics = evaluateRegression(model, features, targets)
  assert metrics["mse"] < 1e-10
  assert metrics["r2"] > 0.99


def testPredictSharesReturnsScalar():
  features = np.array([[100.0, 5.0], [200.0, 10.0]])
  targets = np.array([1000.0, 2000.0])
  model = fitLinearRegression(features, targets)
  prediction = predictShares(model, 150.0, 7.5)
  assert isinstance(prediction, float)


def testRunPipelineOnRealData():
  result = runPipeline(showPlots=False)
  assert result["model"].coef_.shape == (2,)
  assert result["metrics"]["mse"] > 0
  assert -1.0 <= result["metrics"]["r2"] <= 1.0
  assert len(result["filtered"]) <= len(result["dataFrame"])
  assert result["features"].shape[1] == 2


def testRunPipelineExamplePrediction():
  result = runPipeline(showPlots=False)
  expected = predictShares(
    result["model"],
    EXAMPLE_WORD_COUNT,
    EXAMPLE_ENGAGEMENT,
  )
  assert result["examplePrediction"] == expected


def testPipelineConstants():
  assert MAX_WORD_COUNT == 3500
  assert MAX_SHARES == 80000
  assert CSV_FILE.exists()

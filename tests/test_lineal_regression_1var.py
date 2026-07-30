"""Tests para src/ML/LinealRegression1var."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / "src" / "ML" / "LinealRegression1var"
sys.path.insert(0, str(MODULE_DIR))

from linealRegression1var import (  # noqa: E402
  CSV_FILE,
  FEATURE_COLUMN,
  MAX_SHARES,
  MAX_WORD_COUNT,
  TARGET_COLUMN,
  buildRegressionLine,
  evaluateModel,
  extractFeatureAndTarget,
  filterArticles,
  loadArticlesData,
  predictShares,
  trainLinearRegression,
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
    "# Shares",
  }
  assert expected.issubset(set(dataFrame.columns))
  assert len(dataFrame) > 0


def testFilterArticlesAppliesLimits():
  dataFrame = pd.DataFrame(
    {
      FEATURE_COLUMN: [100, 4000, 2000, 500],
      TARGET_COLUMN: [1000, 1000, 90000, 5000],
    }
  )
  filtered = filterArticles(dataFrame)
  assert len(filtered) == 2
  assert filtered[FEATURE_COLUMN].max() <= MAX_WORD_COUNT
  assert filtered[TARGET_COLUMN].max() <= MAX_SHARES


def testFilterArticlesRejectsNegativeLimits():
  dataFrame = pd.DataFrame({FEATURE_COLUMN: [1], TARGET_COLUMN: [1]})
  with pytest.raises(ValueError, match="maxWordCount"):
    filterArticles(dataFrame, maxWordCount=-1)


def testExtractFeatureAndTargetShape():
  dataFrame = pd.DataFrame(
    {
      FEATURE_COLUMN: [100.0, 200.0],
      TARGET_COLUMN: [1000.0, 2000.0],
      "other": [1, 2],
    }
  )
  features, target = extractFeatureAndTarget(dataFrame)
  assert list(features.columns) == [FEATURE_COLUMN]
  assert list(target.columns) == [TARGET_COLUMN]
  assert features.shape == (2, 1)
  assert target.shape == (2, 1)


def testTrainPredictAndEvaluatePerfectLine():
  features = pd.DataFrame({FEATURE_COLUMN: [1.0, 2.0, 3.0, 4.0]})
  target = pd.DataFrame({TARGET_COLUMN: [2.0, 4.0, 6.0, 8.0]})
  model = trainLinearRegression(features, target)
  predictions = predictShares(model, features)
  metrics = evaluateModel(target, predictions, model=model, features=features)

  assert predictions.shape[0] == 4
  assert np.allclose(predictions.reshape(-1), [2.0, 4.0, 6.0, 8.0])
  assert metrics["r2Score"] == pytest.approx(1.0)
  assert metrics["meanSquaredError"] == pytest.approx(0.0)
  assert metrics["score"] == pytest.approx(1.0)


def testBuildRegressionLineSlopeAndIntercept():
  features = pd.DataFrame({FEATURE_COLUMN: [0.0, 1.0, 2.0]})
  target = pd.DataFrame({TARGET_COLUMN: [1.0, 3.0, 5.0]})
  model = trainLinearRegression(features, target)
  xLine, yLine = buildRegressionLine(model, 0, 2)
  assert list(xLine) == [0, 1, 2]
  assert np.allclose(yLine, [1.0, 3.0, 5.0])


def testBuildRegressionLineRejectsInvalidRange():
  features = pd.DataFrame({FEATURE_COLUMN: [1.0, 2.0]})
  target = pd.DataFrame({TARGET_COLUMN: [1.0, 2.0]})
  model = trainLinearRegression(features, target)
  with pytest.raises(ValueError, match="xMax"):
    buildRegressionLine(model, 10, 5)


def testPipelineConstants():
  assert FEATURE_COLUMN == "Word count"
  assert TARGET_COLUMN == "# Shares"
  assert MAX_WORD_COUNT == 3500
  assert MAX_SHARES == 80000
  assert CSV_FILE.exists()

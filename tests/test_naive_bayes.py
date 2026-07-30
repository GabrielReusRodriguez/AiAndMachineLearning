"""Tests para src/ML/NaiveBayes."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE_DIR = Path(__file__).resolve().parents[1] / "src" / "ML" / "NaiveBayes"
sys.path.insert(0, str(MODULE_DIR))

from naiveBayes import (  # noqa: E402
  CSV_FILE,
  N_BEST_COLUMNS,
  RANDOM_STATE,
  TEST_SIZE,
  addDerivedColumns,
  fitGaussianNB,
  loadComprarAlquilarData,
  predictComprar,
  scoreModel,
  selectBestFeatures,
  splitFeaturesTarget,
  splitTrainTest,
)


def testLoadComprarAlquilarDataHasExpectedColumns():
  dataFrame = loadComprarAlquilarData(CSV_FILE)
  expected = {
    "ingresos",
    "gastos_comunes",
    "pago_coche",
    "gastos_otros",
    "ahorros",
    "vivienda",
    "estado_civil",
    "hijos",
    "trabajo",
    "comprar",
  }
  assert expected.issubset(set(dataFrame.columns))
  assert len(dataFrame) == 202


def testAddDerivedColumnsComputesGastosAndInversion():
  dataFrame = pd.DataFrame(
    {
      "gastos_comunes": [100, 200],
      "gastos_otros": [50, 30],
      "pago_coche": [25, 70],
      "vivienda": [300000, 400000],
      "ahorros": [50000, 80000],
    }
  )
  result = addDerivedColumns(dataFrame)
  assert list(result["gastos"]) == [175, 300]
  assert list(result["inversion"]) == [250000, 320000]
  # No modifica el original
  assert "gastos" not in dataFrame.columns


def testSplitFeaturesTargetShape():
  dataFrame = addDerivedColumns(loadComprarAlquilarData())
  features, target = splitFeaturesTarget(dataFrame)
  assert features.shape == (202, 11)
  assert len(target) == 202
  assert set(target.unique()) == {0, 1}


def testSelectBestFeaturesReturnsExpectedColumnsOnRealData():
  dataFrame = addDerivedColumns(loadComprarAlquilarData())
  features, target = splitFeaturesTarget(dataFrame)
  selectedColumns, _, transformed = selectBestFeatures(features, target)
  assert len(selectedColumns) == N_BEST_COLUMNS
  assert selectedColumns == [
    "ingresos",
    "ahorros",
    "hijos",
    "trabajo",
    "inversion",
  ]
  assert transformed.shape == (202, N_BEST_COLUMNS)


def testSelectBestFeaturesRejectsInvalidNBest():
  features = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
  target = pd.Series([0, 1])
  with pytest.raises(ValueError, match="nBest"):
    selectBestFeatures(features, target, nBest=0)
  with pytest.raises(ValueError, match="nBest"):
    selectBestFeatures(features, target, nBest=3)


def testSplitTrainTestSizesAndReproducibility():
  dataFrame = addDerivedColumns(loadComprarAlquilarData())
  trainA, testA = splitTrainTest(dataFrame)
  trainB, testB = splitTrainTest(dataFrame)
  assert len(trainA) == 161
  assert len(testA) == 41
  assert trainA.equals(trainB)
  assert testA.equals(testB)


def testSplitTrainTestRejectsInvalidTestSize():
  dataFrame = addDerivedColumns(loadComprarAlquilarData())
  with pytest.raises(ValueError, match="testSize"):
    splitTrainTest(dataFrame, testSize=0.0)
  with pytest.raises(ValueError, match="testSize"):
    splitTrainTest(dataFrame, testSize=1.0)


def testFitPredictAndScoreOnRealPipeline():
  dataFrame = addDerivedColumns(loadComprarAlquilarData())
  features, target = splitFeaturesTarget(dataFrame)
  selectedColumns, _, _ = selectBestFeatures(features, target)
  trainFrame, testFrame = splitTrainTest(dataFrame)

  trainFeatures = trainFrame[selectedColumns].values
  testFeatures = testFrame[selectedColumns].values
  trainTarget = trainFrame["comprar"]
  testTarget = testFrame["comprar"]

  model = fitGaussianNB(trainFeatures, trainTarget)
  predictions = predictComprar(model, testFeatures)

  assert predictions.shape == (41,)
  assert set(predictions).issubset({0, 1})
  assert scoreModel(model, trainFeatures, trainTarget) == pytest.approx(0.87, abs=0.01)
  assert scoreModel(model, testFeatures, testTarget) == pytest.approx(0.90, abs=0.01)


def testPredictSampleProfiles():
  dataFrame = addDerivedColumns(loadComprarAlquilarData())
  features, target = splitFeaturesTarget(dataFrame)
  selectedColumns, _, _ = selectBestFeatures(features, target)
  trainFrame, _ = splitTrainTest(dataFrame)

  model = fitGaussianNB(
    trainFrame[selectedColumns].values,
    trainFrame["comprar"],
  )
  sampleProfiles = np.array(
    [
      [2000, 5000, 0, 5, 200000],
      [6000, 34000, 2, 5, 320000],
    ]
  )
  assert predictComprar(model, sampleProfiles).tolist() == [0, 1]


def testPipelineConstants():
  assert N_BEST_COLUMNS == 5
  assert TEST_SIZE == 0.2
  assert RANDOM_STATE == 6
  assert CSV_FILE.exists()

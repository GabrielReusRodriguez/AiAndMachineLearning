"""Tests para src/AI/StandAlone/ConvNeuralTensorFlow."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_DIR = (
  Path(__file__).resolve().parents[1]
  / "src"
  / "AI"
  / "StandAlone"
  / "ConvNeuralTensorFlow"
  / "src"
)
sys.path.insert(0, str(MODULE_DIR))

HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None
pytestmark = pytest.mark.skipif(
  not HAS_TENSORFLOW,
  reason="tensorflow no está instalado",
)

if HAS_TENSORFLOW:
  from convNeuralTensorFlow import Modelo  # noqa: E402


def testPrepareDataNormalizesAndAddsChannel():
  raw = np.full((4, 28, 28), 255, dtype=np.uint8)
  prepared = Modelo.prepareData(raw)
  assert prepared.shape == (4, 28, 28, 1)
  assert prepared.dtype == np.float32
  np.testing.assert_allclose(prepared, 1.0)


def testPrepareDataKeepsExistingChannel():
  raw = np.zeros((2, 28, 28, 1), dtype=np.float32)
  prepared = Modelo.prepareData(raw)
  assert prepared.shape == (2, 28, 28, 1)


def testBuildCompileAndPredictShape():
  model = Modelo()
  model.compile()
  trainImgs = np.random.randint(0, 256, size=(32, 28, 28), dtype=np.uint8)
  trainLabels = np.random.randint(0, 10, size=(32,), dtype=np.int32)
  model.train(trainImgs, trainLabels, epochs=2)

  sample = trainImgs[0:1]
  prediction = model.predict(sample)
  assert prediction.shape == (1, 10)
  assert float(np.sum(prediction[0])) == pytest.approx(1.0, abs=1e-5)


def testPredictUsesNormalizedInput():
  model = Modelo()
  model.compile()
  trainImgs = np.random.randint(0, 256, size=(16, 28, 28), dtype=np.uint8)
  trainLabels = np.random.randint(0, 10, size=(16,), dtype=np.int32)
  model.train(trainImgs, trainLabels, epochs=1)

  rawSample = np.full((1, 28, 28), 255, dtype=np.uint8)
  viaApi = model.predict(rawSample)
  viaNorm = model.model.predict(Modelo.prepareData(rawSample))
  viaRaw = model.model.predict(rawSample.astype(np.float32)[..., np.newaxis])

  np.testing.assert_allclose(viaApi, viaNorm, rtol=1e-5, atol=1e-5)
  assert not np.allclose(viaApi, viaRaw, rtol=1e-3, atol=1e-3)


def testSaveAndLoad(tmp_path):
  modelPath = tmp_path / "cnn.keras"
  model = Modelo()
  model.compile()
  trainImgs = np.random.randint(0, 256, size=(16, 28, 28), dtype=np.uint8)
  trainLabels = np.random.randint(0, 10, size=(16,), dtype=np.int32)
  model.train(trainImgs, trainLabels, epochs=1)
  model.save(str(modelPath))
  assert modelPath.is_file()

  loaded = Modelo()
  loaded.load(str(modelPath))
  sample = trainImgs[0:1]
  original = model.predict(sample)
  restored = loaded.predict(sample)
  np.testing.assert_allclose(original, restored, rtol=1e-5, atol=1e-5)


def testEvaluateReturnsLossAndAccuracy():
  model = Modelo()
  model.compile()
  imgs = np.random.randint(0, 256, size=(16, 28, 28), dtype=np.uint8)
  labels = np.random.randint(0, 10, size=(16,), dtype=np.int32)
  model.train(imgs, labels, epochs=1)
  evaluation = model.evaluate(imgs, labels)
  assert len(evaluation) == 2

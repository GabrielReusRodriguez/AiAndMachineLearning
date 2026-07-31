"""Tests para src/AI/StandAlone/FirstNeuralTensorFlow."""

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
  / "FirstNeuralTensorFlow"
  / "src"
)
sys.path.insert(0, str(MODULE_DIR))

HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None
pytestmark = pytest.mark.skipif(
  not HAS_TENSORFLOW,
  reason="tensorflow no está instalado",
)

if HAS_TENSORFLOW:
  from cloth_detector import Modelo as ClothModelo  # noqa: E402
  from intro import Modelo as IntroModelo  # noqa: E402
  from intro import createTrainDataset  # noqa: E402


def testCreateTrainDatasetShapes():
  xs, ys = createTrainDataset(maxTrain=50)
  assert xs.shape == (50,)
  assert ys.shape == (50,)
  assert xs.dtype == float
  assert ys.dtype == float
  np.testing.assert_allclose(ys, 2.0 * xs - 1.0)


def testIntroTrainAndPredictApproximatesFormula():
  import tensorflow as tf

  model = IntroModelo()
  model.new()
  model.compila(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01))
  xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
  ys = 2.0 * xs - 1.0
  model.entrena(xs=xs, ys=ys, epochs=500, callbacks=[])
  prediction = model.prediccion(np.array([10.0], dtype=float))
  assert prediction.shape == (1, 1)
  assert float(prediction[0][0]) == pytest.approx(19.0, abs=1.0)


def testIntroSaveAndLoad(tmp_path):
  modelPath = tmp_path / "model.keras"
  model = IntroModelo()
  model.new()
  model.compila()
  xs = np.linspace(-5.0, 5.0, 21, dtype=float)
  ys = 2.0 * xs - 1.0
  model.entrena(xs=xs, ys=ys, epochs=50)
  model.save(str(modelPath))
  assert modelPath.is_file()

  loaded = IntroModelo()
  loaded.load(str(modelPath))
  original = model.prediccion(np.array([7.0], dtype=float))
  restored = loaded.prediccion(np.array([7.0], dtype=float))
  np.testing.assert_allclose(original, restored, rtol=1e-5, atol=1e-5)


def testClothDetectorBuildCompileAndPredictShape():
  model = ClothModelo()
  model.compile()
  trainImgs = np.random.randint(0, 256, size=(32, 28, 28), dtype=np.uint8)
  trainLabels = np.random.randint(0, 10, size=(32,), dtype=np.int32)
  model.train(trainImgs, trainLabels, epochs=2)

  sample = trainImgs[0:1]
  prediction = model.predict(sample)
  assert prediction.shape == (1, 10)
  assert float(np.sum(prediction[0])) == pytest.approx(1.0, abs=1e-5)


def testClothDetectorPredictUsesNormalizedInput():
  model = ClothModelo()
  model.compile()
  trainImgs = np.random.randint(0, 256, size=(16, 28, 28), dtype=np.uint8)
  trainLabels = np.random.randint(0, 10, size=(16,), dtype=np.int32)
  model.train(trainImgs, trainLabels, epochs=1)

  rawSample = np.full((1, 28, 28), 255, dtype=np.uint8)
  viaApi = model.predict(rawSample)
  viaNorm = model.model.predict(rawSample.astype(np.float32) / 255.0)
  viaRaw = model.model.predict(rawSample.astype(np.float32))

  np.testing.assert_allclose(viaApi, viaNorm, rtol=1e-5, atol=1e-5)
  assert not np.allclose(viaApi, viaRaw, rtol=1e-3, atol=1e-3)

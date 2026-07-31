"""Tests para src/AI/Colab/ConvNeuralTensorFlow_TransferLearning."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

MODULE_DIR = (
  Path(__file__).resolve().parents[1]
  / "src"
  / "AI"
  / "Colab"
  / "ConvNeuralTensorFlow_TransferLearning"
  / "src"
)
sys.path.insert(0, str(MODULE_DIR))

HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None
pytestmark = pytest.mark.skipif(
  not HAS_TENSORFLOW,
  reason="tensorflow no está instalado",
)

if HAS_TENSORFLOW:
  from convNeuralTensorFlowTransferLearning import (  # noqa: E402
    Modelo,
    Stopper,
    createTestGenerator,
    createTrainGenerator,
    splitData,
  )


def _writeTinyJpeg(path: Path, color=(120, 80, 40)):
  Image.new("RGB", (32, 32), color=color).save(path, format="JPEG")


def _makeCatsDogsTree(tmpPath: Path, imagesPerClass=6):
  """Crea un mini dataset cats/dogs con train/test listos o solo origen."""
  sourceCat = tmpPath / "PetImages" / "Cat"
  sourceDog = tmpPath / "PetImages" / "Dog"
  sourceCat.mkdir(parents=True)
  sourceDog.mkdir(parents=True)

  for index in range(imagesPerClass):
    _writeTinyJpeg(sourceCat / f"cat_{index}.jpg", color=(200, 50, 50))
    _writeTinyJpeg(sourceDog / f"dog_{index}.jpg", color=(50, 50, 200))

  # Fichero vacío a ignorar
  (sourceCat / "empty.jpg").write_bytes(b"")

  trainRoot = tmpPath / "cats-v-dogs" / "train"
  testRoot = tmpPath / "cats-v-dogs" / "test"
  for className in ("cats", "dogs"):
    (trainRoot / className).mkdir(parents=True)
    (testRoot / className).mkdir(parents=True)

  return sourceCat, sourceDog, trainRoot, testRoot


def testSplitDataIgnoresEmptyAndRespectsRatio(tmp_path):
  sourceCat, _, trainRoot, testRoot = _makeCatsDogsTree(tmp_path, imagesPerClass=10)
  trainCount, testCount = splitData(
    str(sourceCat),
    str(trainRoot / "cats"),
    str(testRoot / "cats"),
    0.8,
  )
  assert trainCount == 8
  assert testCount == 2
  assert len(list((trainRoot / "cats").iterdir())) == 8
  assert len(list((testRoot / "cats").iterdir())) == 2
  assert not (trainRoot / "cats" / "empty.jpg").exists()


def testStopperStopsWhenAccuracyExceeded():
  stopper = Stopper(accuracyThreshold=0.9)
  fakeModel = type("M", (), {"stop_training": False})()
  stopper.set_model(fakeModel)
  stopper.on_epoch_end(0, {"accuracy": 0.95})
  assert fakeModel.stop_training is True


def testStopperIgnoresMissingMetric():
  stopper = Stopper(accuracyThreshold=0.9)
  fakeModel = type("M", (), {"stop_training": False})()
  stopper.set_model(fakeModel)
  stopper.on_epoch_end(0, {})
  assert fakeModel.stop_training is False


def testBuildCompileAndPredictShape():
  model = Modelo()
  model.define(weights=None)
  model.compile()

  sample = np.random.rand(2, 150, 150, 3).astype(np.float32)
  prediction = model.predict(sample)
  assert prediction.shape == (2, 1)
  assert np.all(prediction >= 0.0)
  assert np.all(prediction <= 1.0)


def testTrainWithTinyGenerators(tmp_path):
  sourceCat, sourceDog, trainRoot, testRoot = _makeCatsDogsTree(
    tmp_path, imagesPerClass=4
  )
  splitData(str(sourceCat), str(trainRoot / "cats"), str(testRoot / "cats"), 0.75)
  splitData(str(sourceDog), str(trainRoot / "dogs"), str(testRoot / "dogs"), 0.75)

  trainGenerator = createTrainGenerator(str(trainRoot), batchSize=2)
  testGenerator = createTestGenerator(str(testRoot), batchSize=2)

  model = Modelo()
  model.define(weights=None)
  model.compile()
  history = model.train(
    trainGenerator,
    validationData=testGenerator,
    epochs=1,
    verbose=0,
  )
  assert "accuracy" in history.history or "acc" in history.history
  evaluation = model.evaluate(testGenerator)
  assert len(evaluation) == 2


def testSaveAndLoad(tmp_path):
  modelPath = tmp_path / "transfer.keras"
  model = Modelo()
  model.define(weights=None)
  model.compile()
  model.save(str(modelPath))
  assert modelPath.is_file()

  loaded = Modelo()
  loaded.load(str(modelPath))
  sample = np.random.rand(1, 150, 150, 3).astype(np.float32)
  original = model.predict(sample)
  restored = loaded.predict(sample)
  np.testing.assert_allclose(original, restored, rtol=1e-5, atol=1e-5)

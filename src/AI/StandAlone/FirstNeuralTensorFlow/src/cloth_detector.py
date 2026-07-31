#!/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

DEFAULT_EPOCHS = 50
ACCURACY_STOP_THRESHOLD = 0.95
NUM_DEMO_PREDICTIONS = 3


class StopTrainCallback(tf.keras.callbacks.Callback):
  """Detiene el entrenamiento al superar el umbral de accuracy."""

  def __init__(self, accuracyThreshold=ACCURACY_STOP_THRESHOLD):
    super().__init__()
    self.accuracyThreshold = accuracyThreshold

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    accuracy = logs.get("accuracy")
    if accuracy is not None and accuracy > self.accuracyThreshold:
      print(
        f"\nSe alcanza el {self.accuracyThreshold * 100:.0f}% de accuracy "
        "por lo que paramos el entrenamiento"
      )
      self.model.stop_training = True


class Modelo:
  """MLP para clasificación Fashion-MNIST (28x28 → 10 clases)."""

  def __init__(self):
    # Flatten → Dense(128, relu) → Dense(10, softmax)
    self.model = tf.keras.models.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation=tf.nn.relu),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax),
    ])

  def compile(self):
    self.model.compile(
      optimizer="adam",
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"],
    )

  def train(self, trainImgs, trainLabels, epochs=DEFAULT_EPOCHS):
    trainImgsNorm = trainImgs / 255.0
    self.model.fit(
      trainImgsNorm,
      trainLabels,
      epochs=epochs,
      callbacks=[StopTrainCallback()],
    )

  def evaluate(self, testImgs, testLabels):
    testImgsNorm = testImgs / 255.0
    return self.model.evaluate(testImgsNorm, testLabels)

  def predict(self, testImg):
    testImgNorm = testImg / 255.0
    return self.model.predict(testImgNorm)


if __name__ == "__main__":
  dataMnist = tf.keras.datasets.fashion_mnist
  (trainImgs, trainLabels), (testImgs, testLabels) = dataMnist.load_data()

  modelo = Modelo()
  modelo.compile()
  modelo.train(trainImgs, trainLabels)
  modelo.evaluate(testImgs, testLabels)

  for i in range(NUM_DEMO_PREDICTIONS):
    print(f"PREDICCION {i}: {modelo.predict(testImgs[i:i + 1])}")

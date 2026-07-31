#!/bin/env python3
# -*- coding: utf-8 -*-

"""CNN con convolución y pooling para clasificar Fashion-MNIST.

Las capas Conv2D + MaxPooling2D ayudan a reconocer patrones locales
independientemente de la posición del objeto en la imagen.
"""

import numpy as np
import tensorflow as tf

DEFAULT_EPOCHS = 50
ACCURACY_STOP_THRESHOLD = 0.9
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
  """CNN para clasificación Fashion-MNIST (28x28x1 → 10 clases)."""

  def __init__(self):
    self.model = None

  def define(self):
    """Define la red: 2 bloques Conv+Pool, Flatten y 2 Dense."""
    # Conv2D + MaxPooling detectan rasgos locales; Dense clasifica.
    # ReLU anula salidas negativas; softmax da probabilidad por clase.
    self.model = tf.keras.models.Sequential([
      tf.keras.Input(shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
      tf.keras.layers.MaxPooling2D(2, 2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation="relu"),
      tf.keras.layers.Dense(10, activation="softmax"),
    ])

  def compile(self):
    if self.model is None:
      self.define()
    self.model.compile(
      optimizer="adam",
      loss="sparse_categorical_crossentropy",
      metrics=["accuracy"],
    )

  def train(self, trainImgs, trainLabels, epochs=DEFAULT_EPOCHS):
    trainImgsPrepared = Modelo.prepareData(trainImgs)
    self.model.fit(
      trainImgsPrepared,
      trainLabels,
      epochs=epochs,
      callbacks=[StopTrainCallback()],
    )

  def save(self, filename: str):
    self.model.save(filename)

  def load(self, filename: str):
    self.model = tf.keras.models.load_model(filename)

  def predict(self, testImg):
    testImgPrepared = Modelo.prepareData(testImg)
    return self.model.predict(testImgPrepared)

  def evaluate(self, testImgs, testLabels):
    testImgsPrepared = Modelo.prepareData(testImgs)
    return self.model.evaluate(testImgsPrepared, testLabels)

  @staticmethod
  def prepareData(data):
    """Normaliza a [0, 1] y asegura canal de profundidad (N, 28, 28, 1)."""
    prepared = data.astype(np.float32) / 255.0
    if prepared.ndim == 3:
      prepared = np.expand_dims(prepared, axis=-1)
    return prepared


if __name__ == "__main__":
  dataMnist = tf.keras.datasets.fashion_mnist
  (trainImgs, trainLabels), (testImgs, testLabels) = dataMnist.load_data()

  modelo = Modelo()
  modelo.define()
  modelo.compile()
  modelo.train(trainImgs, trainLabels)
  evaluacion = modelo.evaluate(testImgs, testLabels)
  print(f"EVALUACION: {evaluacion}")

  for i in range(NUM_DEMO_PREDICTIONS):
    print(f"PREDICCION {i}: {modelo.predict(testImgs[i:i + 1])}")

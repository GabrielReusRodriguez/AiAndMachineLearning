#!/bin/env python3
# -*- coding: utf-8 -*-

import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

DEFAULT_EPOCHS = 50000
DEFAULT_MODEL_PATH = "model.keras"
MAE_STOP_THRESHOLD = 1.0


class StopTrainingCallback(tf.keras.callbacks.Callback):
  """Detiene el entrenamiento cuando el MAE baja del umbral."""

  def __init__(self, maeThreshold=MAE_STOP_THRESHOLD):
    super().__init__()
    self.maeThreshold = maeThreshold

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    mae = logs.get("mae")
    if mae is not None and mae < self.maeThreshold:
      print(
        f"Se alcanzó un error medio absoluto menor a {self.maeThreshold}, "
        "paramos el entrenamiento"
      )
      self.model.stop_training = True


class Modelo:
  """Red mínima Dense(1) para aprender y = 2x - 1."""

  def __init__(self):
    self.model = None

  def new(self):
    self.model = Sequential([
      tf.keras.Input(shape=(1,)),
      Dense(units=1),
    ])

  def save(self, path=DEFAULT_MODEL_PATH):
    self.model.save(path)

  def load(self, path=DEFAULT_MODEL_PATH):
    self.model = tf.keras.models.load_model(path)

  def compila(self, optimizer="adam"):
    self.model.compile(
      optimizer=optimizer,
      loss="mean_squared_error",
      metrics=["mae"],
    )

  def entrena(
    self,
    xs: np.ndarray,
    ys: np.ndarray,
    epochs=DEFAULT_EPOCHS,
    callbacks=None,
  ):
    if callbacks is None:
      callbacks = [StopTrainingCallback()]
    self.model.fit(xs, ys, epochs=epochs, callbacks=callbacks)

  def prediccion(self, entrada: np.ndarray):
    return self.model.predict(entrada)


def createTrainDataset(maxTrain=100):
  x = []
  y = []
  for _ in range(maxTrain):
    value = random.randint(-1000, 1000)
    x.append(float(value))
    y.append(2.0 * float(value) - 1.0)
  xs = np.array(x, dtype=float)
  ys = np.array(y, dtype=float)
  return xs, ys


if __name__ == "__main__":
  print("Hola mundo!")
  model = Modelo()
  model.new()
  model.compila()
  xs, ys = createTrainDataset()
  model.entrena(xs=xs, ys=ys)
  model.save()
  for _ in range(20):
    entrada = random.randint(5, 1000)
    print(
      f" input :{entrada}\tprediccion : "
      f"{model.prediccion(np.array([entrada], dtype=float))}"
    )
  print("Adios mundo!")

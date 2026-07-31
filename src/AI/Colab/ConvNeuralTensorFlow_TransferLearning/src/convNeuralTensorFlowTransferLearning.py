#!/bin/env python3
# -*- coding: utf-8 -*-

"""Transfer learning con InceptionV3 (cats vs dogs).

Congela el backbone preentrenado en ImageNet, toma la salida de `mixed7`
y añade una cabeza densa para clasificación binaria.
"""

import os
import random
from shutil import copyfile

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DEFAULT_EPOCHS = 20
STOP_ACCURACY = 0.9
TRAINING_SIZE = 0.9
INPUT_SHAPE = (150, 150, 3)
TARGET_SIZE = (150, 150)
DENSE_UNITS = 1024
LEARNING_RATE = 0.0001
LAST_LAYER_NAME = "mixed7"
BATCH_SIZE = 100


class Stopper(tf.keras.callbacks.Callback):
  """Detiene el entrenamiento al superar el umbral de accuracy."""

  def __init__(self, accuracyThreshold=STOP_ACCURACY):
    super().__init__()
    self.accuracyThreshold = accuracyThreshold

  def on_epoch_end(self, epoch, logs=None):
    logs = logs or {}
    accuracy = logs.get("accuracy")
    if accuracy is None:
      accuracy = logs.get("acc")
    if accuracy is not None and accuracy > self.accuracyThreshold:
      print(
        f"\nSe ha superado la precisión deseada {accuracy}"
      )
      self.model.stop_training = True


def splitData(sourceFolder, trainingFolder, testingFolder, splitSize):
  """Divide ficheros no vacíos en carpetas de train/test según `splitSize`."""
  files = []
  for filename in os.listdir(sourceFolder):
    filePath = f"{sourceFolder}/{filename}"
    if os.path.getsize(filePath) > 0:
      files.append(filename)
    else:
      print(f"Se ignora el fichero {filename} de tamano 0.")

  trainingLength = int(len(files) * splitSize)
  shuffled = random.sample(files, len(files))
  trainingFiles = shuffled[0:trainingLength]
  testingFiles = shuffled[trainingLength:]

  for filename in trainingFiles:
    copyfile(f"{sourceFolder}/{filename}", f"{trainingFolder}/{filename}")

  for filename in testingFiles:
    copyfile(f"{sourceFolder}/{filename}", f"{testingFolder}/{filename}")

  return len(trainingFiles), len(testingFiles)


def createTrainGenerator(trainDir, batchSize=BATCH_SIZE, targetSize=TARGET_SIZE):
  """DataGenerator de entrenamiento con data augmentation."""
  trainDatagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
  )
  return trainDatagen.flow_from_directory(
    trainDir,
    batch_size=batchSize,
    class_mode="binary",
    target_size=targetSize,
  )


def createTestGenerator(testDir, batchSize=BATCH_SIZE, targetSize=TARGET_SIZE):
  """DataGenerator de validación/test (solo rescale)."""
  testDatagen = ImageDataGenerator(rescale=1.0 / 255)
  return testDatagen.flow_from_directory(
    testDir,
    batch_size=batchSize,
    class_mode="binary",
    target_size=targetSize,
  )


class Modelo:
  """Clasificador binario sobre InceptionV3 congelado + cabeza densa."""

  def __init__(self):
    self.model = None
    self.preTrainedModel = None

  def define(self, weights="imagenet", lastLayerName=LAST_LAYER_NAME):
    """Construye InceptionV3 + Flatten + Dense + sigmoid."""
    self.preTrainedModel = InceptionV3(
      input_shape=INPUT_SHAPE,
      include_top=False,
      weights=weights,
    )
    for layer in self.preTrainedModel.layers:
      layer.trainable = False

    lastLayer = self.preTrainedModel.get_layer(lastLayerName)
    lastOutput = lastLayer.output
    x = layers.Flatten()(lastOutput)
    x = layers.Dense(DENSE_UNITS, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)
    self.model = Model(self.preTrainedModel.input, x)

  def compile(self):
    if self.model is None:
      self.define()
    self.model.compile(
      optimizer=RMSprop(learning_rate=LEARNING_RATE),
      loss="binary_crossentropy",
      metrics=["accuracy"],
    )

  def train(
    self,
    trainGenerator,
    validationData=None,
    epochs=DEFAULT_EPOCHS,
    verbose="auto",
  ):
    return self.model.fit(
      trainGenerator,
      validation_data=validationData,
      epochs=epochs,
      verbose=verbose,
      callbacks=[Stopper()],
    )

  def predict(self, images):
    return self.model.predict(images)

  def evaluate(self, dataGenerator):
    return self.model.evaluate(dataGenerator)

  def save(self, filename: str):
    self.model.save(filename)

  def load(self, filename: str):
    self.model = tf.keras.models.load_model(filename)

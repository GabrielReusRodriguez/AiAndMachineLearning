#!./venv/bin/python3

# En este ejercicio haremos un modelo de regresion logistica que tiuene salida discreta ( y no continua)
# Tenemos en cuenta que los valores etiquetados son 0 para Windows, 1 para Machintosh y 2 para Linux

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb

NAME_DATASET = './data/usuarios_win_mac_lin.csv'

# Cargamos el dataset
dataframe = pd.read_csv(NAME_DATASET)
print(dataframe.head())

# Printamos la informacion estadistica.
print(dataframe.describe())

# Contamos el número de elementos de cada clase.
print(dataframe.groupby('clase').size())

# Hacemos una visualizacion de los datos
dataframe.drop(['clase'], axis = 'columns').hist()

# Printamos más estadisticas
sb.pairplot(dataframe.dropna(), hue = 'clase', height= 4, vars = ['duracion', 'paginas', 'acciones','valor'], kind = 'reg')

plt.show()

# Pasamos a crear el modelo
X= np.array(dataframe.drop(['clase'],axis = 'columns'))
y = np.array(dataframe['clase'])
X.shape

model = linear_model.LogisticRegression()
model.fit(X, y)

# Predecimos.
predictions = model.predict(X)
print(predictions)

print(f"model Score: {model.score(X, y)}")

# Validamos el modelo
validation_size = 0.2
seed = 7
X_train, X_validation, y_train, y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)

name= 'Logistic Regression'
# Kfold se usa dividir los datos en entreno y test para poder usar el validador cruzado.
#kfold = model_selection.KFold(n_splits=10, random_state=seed)
kfold = model_selection.KFold(n_splits=10)
cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)

# Hagamos predicciones.
predictions = model.predict(X_validation)
print(accuracy_score(y_validation, predictions))

# Reporte de resultados del modelo

print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))



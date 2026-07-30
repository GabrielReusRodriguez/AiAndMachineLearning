#!./venv/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import  linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,  r2_score

from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler


DATA_LOCATION = './data/articulos_ml.csv'

NUM_RESUME_LINES = 10

# Cargamos el dataFrame.

df_articles = pd.read_csv(
    filepath_or_buffer= DATA_LOCATION
    )

#print(df_articles.head(NUM_RESUME_LINES))
#print(df_articles.describe())

# Histograma de los datos.
#fig, ax = plt.subplots(1,1, figsize = (20,10))
#df_articles.drop(['Title', 'url', 'Elapsed days'], axis = 'columns').hist(ax= ax)
#df_articles.drop(['Title', 'url', 'Elapsed days'], axis = 'columns').hist()
# Nos quedamos con la mayoria de datos


df_filtered_data = df_articles[(df_articles['Word count'] <= 3500) & (df_articles['# Shares'] <= 80000)]

# Generamos una tercera variable que es la suma de enlaces , comentarios e imagenes.

suma = (df_filtered_data['# of Links'] + df_filtered_data['# of comments'].fillna(0) + df_filtered_data['# Images video'])

# Creamos el nuevo dataFrame.
dataX2 = pd.DataFrame()
dataX2["Word count"] = df_filtered_data["Word count"]
dataX2["suma"] = suma

# Generamos el juego de datos de entrenamiento
XY_train = np.array(dataX2)
# Genero los resultados de las muestras
z_train =  df_filtered_data['# Shares'].values

# Una vez generado el set de entenamiento, generamos el regresos lineal de 2 dimensiones.
regr2 = linear_model.LinearRegression()
# Entrenamos el regresor.
regr2.fit(XY_train, z_train)
# Hacemos la predicción.
z_pred = regr2.predict(XY_train)
# Printamos el modelo.
print(f"Coeficientes: \n {regr2.coef_}")

# Calculamos el rendimiento
# Error cuadrativo medio, cuanto menos , mejor.
print(f"Mean squared error : \n {mean_squared_error(z_train, z_pred):.2f}")
# Cuanto mas cerca de 1 mejor.
print(f"Variance score : \n {r2_score(z_train, z_pred):.2f}")

# Prediccion 
print(f"Num shares previsto: {int(regr2.predict([[2000,10+4+6]]))}")

# Visualizamos los datos en 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
#ax = Axes3D(fig)

# Creamos la malla sobre la que haremos el grafico del plano.

xx, yy = np.meshgrid(np.linspace(0, 3500, num = 10), np.linspace(0,60,num=10))

# Calculamos los valores del plano para los puntos x e y
nuevoX = (regr2.coef_[0] * xx)
nuevoY = (regr2.coef_[1] * yy)

# Calculamos los valores para la coordenada z.

z = (nuevoX + nuevoY + regr2.intercept_)
#z = (regr2.coef_[0] * xx  + regr2.coef_[1] * yy + regr2.intercept_)

# Tenemos generado el plano ya que tenemos x, y,z asi que lo graficamos.
# Esta superficie es la  generaliza la nube de puntos.
ax.plot_surface(xx,yy,z, alpha = 0.2, cmap = 'hot')

# graficamos los puntos en 3D

ax.scatter(XY_train[:,0], XY_train[:,1], z_train, c='blue', s=30 )

#Situamos la camara  (azim es el angulo)
ax.view_init(elev=30., azim = 65)

ax.set_xlabel('Cantidad de Palabras')
ax.set_ylabel('Cantidad de enlaces, comentarios e imagenes')
ax.set_zlabel('Compartido en redes')
ax.set_title('Regresion lineal con multiples variables')

plt.show()

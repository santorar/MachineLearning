import numpy as np
from sklearn.datasets import load_iris
from pprint import pprint
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

datos = load_iris()
# pprint(datos)

clasification = datos.target_names
print(clasification)
features = datos.feature_names
X = datos.data
y = datos.target
y = y.reshape(-1 ,1)

# Grafica de dispersion de los datos
# plt.figure(figsize=(18,8),dpi=100)   #set the canvas size for visibility

# plt.scatter(X.T[0],X.T[2])   #over here I use the T ndarray method to transpose the data then get columns at index 0 and 2
# plt.title('IRIS Petal and sepal length', fontsize=20) # set the title of the plot and adjust my font size for readability

# #then we set the label (just to be obvious)
# plt.ylabel('Petal Length') 
# plt.xlabel('sepal length')



lr = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

iris_model = lr.fit(X_train, y_train)

y_pred = iris_model.predict(X_test)
y_error = np.abs(y_test.flatten() - y_pred.flatten())

#Grafica de barras de error
# plt.errorbar(range(len(y_test)), y_test.flatten(), yerr=y_error, fmt='^k', ecolor='red')
# plt.show()

# Grafica de la regresion Lineal
# plt.figure(figsize=(8,6))
# plt.scatter(y_test, y_pred, alpha=0.7, color="blue", label="Predicciones")
# plt.plot([y_test.min(), y_test.max()],
#          [y_test.min(), y_test.max()],
#          'r--', lw=2, label="Línea ideal (y = y_pred)")
# plt.yticks([0,1,2], clasification)
# plt.xlabel("Valores reales (y_test)")
# plt.ylabel("Valores predichos (y_pred)")
# plt.title("Regresión lineal: valores reales vs. predichos")
# plt.legend()
# plt.show()

y_pred_classes = np.abs(np.round(y_pred))
print(y_pred_classes)

print(classification_report(y_test, y_pred_classes))

#Matriz de confusion
# cm = confusion_matrix(y_test, y_pred_classes)

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=datos.target_names)
# disp.plot(cmap="Blues")
# plt.title("Matriz de confusión")
# plt.show()

# Grafica de pesos de variables
# coef = iris_model.coef_.flatten()
# plt.barh(datos.feature_names, coef)
# plt.title("Importancia de variables según el modelo")
# plt.xlabel("Peso (coeficiente)")
# plt.show()
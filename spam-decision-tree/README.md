# Arbol de Decisión 

En esta práctica se busca clasificar los correos por spam y ham, haciendo uso del dataset `prueba_dataset.csv`, este dataset fue creado a apartir de datos sintéticos con features que describieran al menos una característica de un correo con spam.

Este algoritmo utiliza las librerias de `sklearn`, `matplotlib`, `pandas`, `numpy` y `scipy`, todas las dependencias se encuentran dentro del `Pipfile.txt` en la carperta general, y solo es necesario ejecutar el comando `pipenv install` para descargarlas.

## Funcionamiento

Este algoritmo toma el dataset, luego aplica una serie de pasos para tratar los datos del mismo de la manera más adecuada, todo esto s ehace con el fin de preparar los datos para que cuando ingresen en el modelo esten adecuados para su fin.

Luego de estandarizar y adecuar el dataset este se divide en una parte para entrenamiento y una para test, en este caso se tomó el 30% de los datos para hacer pruebas y el 70% para el entrenamiento.

Finalmente usando la libreria de `sklearn` se le llama a la función de `DecissionTreeClasifier` para que el código entienda con cual de los módelos se trabajará, y se entrena.

La ejecución de lo anterior se repite 50 veces y de estas se selecciona el que tenga las mejores características, con `matplotlib` se grafican los 50 resultados de cada modelo, esto se hace sobre el f1 score, la precisión del modelo y el z score de cada iteración.

Por último se elige el modelo que mejor resultados halla dado y se ve el arbol obtenido mediante la gráfica que nos brinda este.
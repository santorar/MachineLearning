# Clasificación de Spam Utilizando Árboles de Decisión.
 
 Integrantes: *David Santiago Sierra Fernández, Oscar Felipe Valcárcel Peralta 
 
 **Link del Repositorio: https://github.com/santorar/MachineLearning/tree/main/spam-decision-tree. 
 
## 1. Introducción 

El correo no deseado (spam) representa un problema persistente en la comunicación digital. Esta práctica aborda el desarrollo de un clasificador de spam basado en un modelo de árbol de decisión se hará uso de la libreria `sklearn` de python.

### 1.1. Objetivo

El objetivo principal es alcanzar el máximo rendimiento posible, medido a través de métricas de clasificación clave como la puntuación F1, la precisión (accuracy) y el z-score.

## 2. Preparación del Modelo

El modelo que se va a utiizar es
### 2.1. Preparación de los datos

Como primer paso llamamos el dataset el cual en esta practica es llamado `prueba_dataset.csv`,  además para evitar valores null se le asignan strings vacíos a las features de `palabras_clave` y `links` esto para que no se rompa el modelo y pueda tratar estos datos.

Luego de esto asignamos las entradas del modelo (X) y las salidas deseadas (y), una vez separadas entradas y salidas se separan los features en 4 subgrupos, los cuales son features de texto, categóricos, numéricos y boleanos. Finalmente se pasan las features de palabras clave y links por la función `TfidVectoorizer` la cual convierte en vectores numéricos las cadenas de texto dando un valor más alto a aquellas palabras que son raras dentro del contexto de los datos, para las features categóricas se pasan por `oneHotEncoder`, el cual da valores numéricos a las diferentes categorías de un conjunto de datos, para las variables numéricas se aplica la función `StandardScaler`, el cual estandariza los datos para evitar valores demasiado grandes que compliquen su uso en el entrenamiento.

### 2.2. Creación del Modelo

Para la creación del modelo primero se separó el dataset completo en dos partes una de entrenamiento y otra de pruebas, en este caso se tomo un 30% del dataset para hacer pruebas y el resto para el entrenamiento. Luego se declaró que este modelo es un arbol de decisión haciendo uso de la función de `sklearn` llamada `DecissionTreeClasifier`, el cual al pasarle los datos de entrenamientoo ya tratados solo es necesario utilizar la función `fit()` este internamente utiliza el índice de gini para tomar las decisiones, esto nos dará el resultado final del entrenamiento.

## 3. Resultados 

Para la medición de la calidad del entrenamiento se utilizarón tres medidas, la precisión (accuracy), el f1 score y finalmente el z score. Además se hizo la ejecución del algoritmo unas 50 veces para determinar cual era la mejor respuesta obtenida por este.

### 3.1. Resultados Generales

En la siguiente gráfica se mostrará los resultados obtenidos de parte de las 50 ejecuciones, donde se verá la precisión del modelo y su f1 score en cada iteración.

![[f1precission.jpeg]]

De esto podemos concluir que en cuanto a la fiabilidad del modelo junto con el dataset son buenos, ya que la mayoría de resultados se muestran por encima del 80%, además de todas las iteraciones el mejor f1 score obtenido fue del 92.6%.

El z score lo que hace es decirnos que tan por encima o por debajo de la media se encuentran los resultados del f1 score, esto quiere decir que nos dará la cantidad de desviaciones estándar que hay desde la media aritmética hasta el resultado obtenido.

![[zscore.jpeg]]

Todo esto realmente se traduce a una sola cosa, la estabilidad del modelo, lo sual a su vez nos dice que tan fiable es, en este caso la mayoría de los resultados se encuentran entre 1 desviación estándar hacia arriba y hacia abajo, sin embargo se ven casos que llegan hasta 2 desviaciones y uno solo que se pasa de las dos.

De todo lo anterior se puede concluir que a pesar de que la mayoría tuvieron precisiones altas, algunas de las respuestas no son las mejores ni las más óptimas.

### 3.2. Árbol Seleccionado

El árbol de decisión con mayor f1 score, precisión y mejor z score es el que se mostrará a continuación:

![[decissiontree.jpeg]]

Este árbol toma decisiones haciendo uso del indice de gini, este empieza tomando la reputación de la IP como punto de partida, donde se clasifican alrededor de 3000 correos directamente como ham, luego vemos que la siguiente feature mas importante son los correos recibidos por el mismo remitente, el cual nos clasifica de 1000 a 2000 correos en dos partes diferentes desde las cuales se empiezan a aplicar decisiones a partir del dominio y de los links incorporados. Desde este punto se aplican las demás features dependiendo de ciertos casos específicos.
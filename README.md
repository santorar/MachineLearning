*Sierra Fernández David Santiago, Valcárcel Peralta Oscar Felipe*
# Identificación de Spam Usando Regresión Logística
## 1. Introducción
Esto es una práctica de machine learning, donde buscámos desarrollar un programa en Python haciendo uso de la librería sklearn que pueda clasificar **spam** o **ham** utilizando el algoritmo de **regresión logística**.

### 1.1 Objetivo
El objetivo principal es entrenar un modelo que tras analizar lod features de un conjunto de mensajes previamente etiquetados, sea capaz de predecir con un grado de certeza aceptable si un nuevo mensaje que no halla visto antes es spam o no.

## 2. Preparación de los datos
### 2.1 Features Usados
1. **Palabras Clave:** texto
2. **Link:** texto
3. **Dominio:** Categorias
4. **Autentificación:** Categorias
5. **Correos recibidos del mismo remitente:** numérico
6. **Reputación IP:** numérico
7. **Participación positiva:** numérico
8. **Blacklisted:** booleano

### 2.2 Features Descartados
1. **Asunto:** Fue descartada porque las palabras clave cumplen con la misma fución de encontrar esas palabras sospechosas que pueden llegar a ser propias del spam.
2. **Cuerpo:** Fue descaratada por que tanto las palabras clave como los links ya cumplen la función de identificar si el contenido es sospechoso.
3. **Remitente:** Se descartó el uso de este feature porque tanto el dominio, la utentificación y la marca de blacklist ya cumplen con ser subcaracterísticas del remitente y al ser más específicas y más fáciles de cuantificar hacen que el remitente quede en un segundo plano.
4. **IP del Remitente:** Esta al igual que con el remitente ya está siendo caracterizada por otros features más específicos.

## 3. Entrenamiento del modelo

## 4. Evaluación del Modelo

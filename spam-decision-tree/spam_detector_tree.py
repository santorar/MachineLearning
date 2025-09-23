import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score, precision_score
import matplotlib.pyplot as plt
import joblib
import numpy as np

data = pd.read_csv("spam-decision-tree/prueba_dataset.csv")
data["palabras_clave"] = data["palabras_clave"].fillna("")
data["links"] = data["links"].fillna("")

#
# Separacion de las variables
#
X = data.drop("spam", axis=1)
y = data["spam"]

# Variables texto plano
text_features = ["palabras_clave", "links"]

# Variables categóricas
cat_features = ["dominio", "autentificacion"]

# Variables numéricas
num_features = ["correos_recibidos_mismo_remitente", "reputacion_ip", "participacion_positiva"]

# Variables booleanas
bool_features = ["blacklisted"]

#
# Preprocesamiento
#

preprocesador = ColumnTransformer(
  transformers = [
    ("keywords", TfidfVectorizer(), "palabras_clave"),
    ("links", TfidfVectorizer(), "links"),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("num", StandardScaler(), num_features),
    ("bool", "passthrough", bool_features)
  ]
)


# Listas para almacenar métricas
f1_scores = []
precision_scores = []
best_f1_score = 0
best_model = None
best_features = None

# Bucle de 50 iteraciones
for i in range(50):
    # Separación de datos con una semilla aleatoria diferente en cada iteración
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)
    
    # Creación y entrenamiento del modelo
    model = Pipeline([
        ("preprocessor", preprocesador),
        ("classifier", DecisionTreeClassifier(max_depth=5))
    ])
    model.fit(X_train, y_train)
    
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Cálculo de métricas
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    
    f1_scores.append(f1)
    precision_scores.append(precision)
    
    # Guardar el mejor modelo
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model = model
        
        # Generar los nombres de las características para el mejor modelo
        feature_names = []
        keywords_vectorizer = best_model.named_steps["preprocessor"].named_transformers_["keywords"]
        feature_names.extend(keywords_vectorizer.get_feature_names_out())

        links_vectorizer = best_model.named_steps["preprocessor"].named_transformers_["links"]
        feature_names.extend(links_vectorizer.get_feature_names_out())

        ohe = best_model.named_steps["preprocessor"].named_transformers_["cat"]
        feature_names.extend(ohe.get_feature_names_out(cat_features))
        
        feature_names.extend(num_features)
        feature_names.extend(bool_features)
        best_features = feature_names

print("Entrenamiento completado en 50 iteraciones.")
print(f"Mejor F1-score: {best_f1_score:.4f}")

fig, ax = plt.subplots(figsize=(15, 8))
bar_width = 0.35
index = np.arange(50)

ax.bar(index, f1_scores, bar_width, label='F1-score', color='skyblue')
ax.bar(index + bar_width, precision_scores, bar_width, label='Precisión', color='orange')

ax.set_xlabel('Iteración de Entrenamiento')
ax.set_ylabel('Puntuación')
ax.set_title('Precisión y F1-score en 50 Iteraciones de Entrenamiento')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(index + 1, rotation=45)
ax.legend()
plt.tight_layout()
plt.show()

if best_model:
    arbol_entrenado = best_model.named_steps["classifier"]
    plt.figure(figsize=(25, 20))
    plot_tree(arbol_entrenado,
              feature_names=best_features,
              class_names=["No Spam", "Spam"],
              filled=True,
              rounded=True,
              fontsize=8)
    plt.title(f"Árbol de Decisión con el Mejor Rendimiento (F1-score: {best_f1_score:.4f})", fontsize=15)
    plt.show()
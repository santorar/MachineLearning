import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import numpy as np

data = pd.read_csv("prueba_dataset.csv")
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

#
# Creación de la inteligencia artificial
#
model = Pipeline([
  ("preprocessor", preprocesador),
  ("classifier", LogisticRegression(max_iter=1000))
])

#
# Entrenamiento
#

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3, random_state=100)
# model = joblib.load("modelo_spam.pk1")
model.fit(X_train, y_train)

#
# Evaluación
#

y_pred = model.predict(X_test)

joblib.dump(model, "modelo_spam_3.pk1")


print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=[False, True])

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Spam", "Spam"])
disp.plot(cmap="Blues")

plt.show()

clf = model.named_steps["classifier"]

vectorizer = model.named_steps["preprocessor"]
feature_names = vectorizer.get_feature_names_out()

coeficientes = clf.coef_[0]

pesos = pd.DataFrame({
    "Variable": feature_names,
    "Peso": coeficientes,
    "Peso_absoluto": np.abs(coeficientes)
}).sort_values("Peso_absoluto", ascending=False)

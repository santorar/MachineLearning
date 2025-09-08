import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = pd.read_csv("correos_dataset.csv")
data["Asunto"] = data["Asunto"].fillna("")
data["Cuerpo"] = data["Cuerpo"].fillna("")
data["palabras_clave"] = data["palabras_clave"].fillna("")
data["links"] = data["links"].fillna("")

#
# Separacion de las variables
#
X = data.drop("spam", axis=1)
y = data["spam"]

# Variables texto plano
text_features = ["Asunto", "Cuerpo", "palabras_clave", "links"]

# Variables categóricas
cat_features = ["remitente", "dominio", "ip_remitente", "autentificacion"]

# Variables numéricas
num_features = ["correos_recibidos_mismo_remitente", "reputacion_ip", "participacion_positiva"]

# Variables booleanas
bool_features = ["blacklisted"]

#
# Preprocesamiento
#

preprocesador = ColumnTransformer(
  transformers = [
    ("text", TfidfVectorizer(), "Asunto"),
    ("cuerpo", TfidfVectorizer(), "Cuerpo"),
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

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)
model.fit(X_train, y_train)

#
# Evaluación
#

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
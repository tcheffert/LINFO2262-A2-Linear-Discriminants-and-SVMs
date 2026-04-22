import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 1. Charger les données
train_df = pd.read_csv("HeartTrain.csv")
X = train_df.drop(columns=['labels'])
y = train_df['labels']

# 2. Prétraitement (Crucial pour comparer les poids !)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Entraîner le SVM Linéaire
# On utilise le modèle simple de la Q1
svm_lin = SVC(kernel='linear')
svm_lin.fit(X_scaled, y)

# 4. Extraire les coefficients (poids)
weights = svm_lin.coef_[0]

# 5. Identifier les 4 plus grands poids en valeur absolue
# On récupère les indices des 4 plus grandes valeurs
indices_top4 = np.argsort(np.abs(weights))[-4:]

# 6. Récupérer les noms des colonnes correspondantes
features_names = X.columns[indices_top4].tolist()

# 7. Formater la réponse
print("Copie-colle ceci :")
print(",".join(features_names))
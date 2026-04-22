import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Chargement des données
train_df = pd.read_csv("./HeartTrain.csv")
test_df = pd.read_csv("./HeartTest.csv")

X_train = train_df.drop(columns=['labels'])
y_train = train_df['labels']

# Préparation de la validation croisée identique à Inginious
kf = KFold(n_splits=10, shuffle=False)
pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])

print("="*70)
print("QUESTION 4: Optimisation des noyaux RBF et Sigmoid")
print("="*70)

# ========== RECHERCHE POUR RBF ==========
print("\n" + "="*70)
print("1. RECHERCHE POUR LE NOYAU RBF")
print("="*70)

#--------- Current best RBF -------#
# RBF kernel parameters
# my_rbf_svm = SVC(kernel='rbf', C=25, gamma='scale') #0.872727
#---------------------------------#

param_rbf = {
    'svm__kernel': ['rbf'],
    'svm__C': [3.539100, 10, 13.538762, 15, 25],
    'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.014636, 0.1, 0.049256, 0.5]
}

print(f"Nombre de combinaisons RBF: {np.prod([len(v) for v in param_rbf.values()])}")
print("Recherche en cours...\n")

grid_rbf = GridSearchCV(pipe, param_rbf, cv=kf, scoring='accuracy', verbose=1, n_jobs=-1)
grid_rbf.fit(X_train, y_train)

print("\n" + "-"*70)
print("RÉSULTATS RBF:")
print("-"*70)
print(f"Meilleurs paramètres: {grid_rbf.best_params_}")
print(f"Meilleure accuracy (CV): {grid_rbf.best_score_:.6f}")

# Top 5 pour RBF
results_rbf = pd.DataFrame(grid_rbf.cv_results_)
top_5_rbf = results_rbf.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
print("\nTop 5 des meilleures combinaisons RBF:")
for idx, row in top_5_rbf.iterrows():
    print(f"  Score: {row['mean_test_score']:.6f} (±{row['std_test_score']:.6f})")
    print(f"  Params: {row['params']}")

best_rbf = grid_rbf.best_params_

# ========== RECHERCHE POUR SIGMOID ==========
print("\n\n" + "="*70)
print("2. RECHERCHE POUR LE NOYAU SIGMOID")
print("="*70)

param_sigmoid = {
    'svm__kernel': ['sigmoid'],
    'svm__C': [50, 60, 70, 75, 80, 90, 100],
    'svm__gamma': ['scale', 'auto', 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 1],
    'svm__coef0': [-1, -0.5, 0, 0.5, 1, 2, 3, 4, 5]
}

print(f"Nombre de combinaisons Sigmoid: {np.prod([len(v) for v in param_sigmoid.values()])}")
print("Recherche en cours...\n")

grid_sigmoid = GridSearchCV(pipe, param_sigmoid, cv=kf, scoring='accuracy', verbose=1, n_jobs=-1)
grid_sigmoid.fit(X_train, y_train)

print("\n" + "-"*70)
print("RÉSULTATS SIGMOID:")
print("-"*70)
print(f"Meilleurs paramètres: {grid_sigmoid.best_params_}")
print(f"Meilleure accuracy (CV): {grid_sigmoid.best_score_:.6f}")

# Top 5 pour Sigmoid
results_sigmoid = pd.DataFrame(grid_sigmoid.cv_results_)
top_5_sigmoid = results_sigmoid.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
print("\nTop 5 des meilleures combinaisons Sigmoid:")
for idx, row in top_5_sigmoid.iterrows():
    print(f"  Score: {row['mean_test_score']:.6f} (±{row['std_test_score']:.6f})")
    print(f"  Params: {row['params']}")

best_sigmoid = grid_sigmoid.best_params_

# ========== RÉSUMÉ FINAL ==========
print("\n\n" + "="*70)
print("RÉSUMÉ FINAL - SOLUTIONS POUR LA SOUMISSION")
print("="*70)

print("\n>>> RBF Kernel <<<")
print(f"Accuracy: {grid_rbf.best_score_:.6f}")
print(f"my_rbf_svm = SVC(kernel='rbf', ", end="")
print(f"C={best_rbf['svm__C']}, ", end="")
if isinstance(best_rbf['svm__gamma'], str):
    print(f"gamma='{best_rbf['svm__gamma']}')")
else:
    print(f"gamma={best_rbf['svm__gamma']})")

print("\n>>> Sigmoid Kernel <<<")
print(f"Accuracy: {grid_sigmoid.best_score_:.6f}")
print(f"my_sigm_svm = SVC(kernel='sigmoid', ", end="")
print(f"C={best_sigmoid['svm__C']}, ", end="")
if isinstance(best_sigmoid['svm__gamma'], str):
    print(f"gamma='{best_sigmoid['svm__gamma']}', ", end="")
else:
    print(f"gamma={best_sigmoid['svm__gamma']}, ", end="")
print(f"coef0={best_sigmoid['svm__coef0']})")

print("\n" + "="*70)

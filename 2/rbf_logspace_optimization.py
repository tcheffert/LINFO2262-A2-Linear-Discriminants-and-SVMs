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
X_test = test_df.drop(columns=['labels'])
y_test = test_df['labels']

# Préparation de la validation croisée identique à Inginious
kf = KFold(n_splits=10, shuffle=False)
pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC())])

# print("="*70)
# print("OPTIMISATION RBF AVEC LOGSPACE")
# print("="*70)
# print(f"Objectif: Battre 0.8727 (CV) et 0.8731 (test)")
# print("="*70)

# ========== RECHERCHE FINE AVEC LOGSPACE ==========
# print("\n" + "="*70)
# print("1. RECHERCHE FINE AVEC LOGSPACE")
# print("="*70)

# Utilisation de logspace pour explorer les paramètres
# C: de 10^0 à 10^3 (1 à 1000)
# gamma: de 10^-4 à 10^0 (0.0001 à 1) + 'scale' et 'auto'
param_rbf_logspace = {
    'svm__kernel': ['rbf'],
    'svm__C': np.logspace(0, 2, 80),  # 80 valeurs entre 1 et 100
    'svm__gamma': ['scale', 'auto'] + list(np.logspace(-4, 0, 100))  # 'scale', 'auto' + 50 valeurs entre 0.001 et 1
}

print(f"Nombre de combinaisons: {len(param_rbf_logspace['svm__C']) * len(param_rbf_logspace['svm__gamma'])}")
print("Plage C:", f"[{param_rbf_logspace['svm__C'][0]:.4f}, {param_rbf_logspace['svm__C'][-1]:.4f}]")
# print("Plage gamma:", f"[{param_rbf_logspace['svm__gamma'][0]:.6f}, {param_rbf_logspace['svm__gamma'][-1]:.6f}]")
print("\nRecherche en cours...\n")

grid_rbf_logspace = GridSearchCV(pipe, param_rbf_logspace, cv=kf, scoring='accuracy', n_jobs=-1)
grid_rbf_logspace.fit(X_train, y_train)

print("\n" + "-"*70)
print("RÉSULTATS LOGSPACE:")
print("-"*70)
print(f"Meilleurs paramètres: {grid_rbf_logspace.best_params_}")
print(f"Meilleure accuracy (CV): {grid_rbf_logspace.best_score_:.6f}")

# Test sur l'ensemble de test
y_pred = grid_rbf_logspace.predict(X_test)
test_accuracy = np.mean(y_pred == y_test)
# print(f"Accuracy sur le test: {test_accuracy:.6f}")

# Top 10 pour RBF
results_rbf = pd.DataFrame(grid_rbf_logspace.cv_results_)
top_10_rbf = results_rbf.nlargest(10, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
# print("\nTop 10 des meilleures combinaisons:")
# print("-"*70)
for idx, row in top_10_rbf.iterrows():
    C_val = row['params']['svm__C']
    gamma_val = row['params']['svm__gamma']
    # print(f"Score CV: {row['mean_test_score']:.6f} (±{row['std_test_score']:.6f})")
    # print(f"  C={C_val:.4f}, gamma={gamma_val:.6f}")

best_rbf = grid_rbf_logspace.best_params_

# ========== RECHERCHE RAFFINÉE AUTOUR DU MEILLEUR ==========
# print("\n\n" + "="*70)
# print("2. RECHERCHE RAFFINÉE AUTOUR DU MEILLEUR")
# print("="*70)

# Créer une grille raffinée autour des meilleurs paramètres
best_C = best_rbf['svm__C']
best_gamma = best_rbf['svm__gamma']

# Recherche dans un voisinage de ±30% autour des meilleurs paramètres
C_min = best_C * 0.7
C_max = best_C * 1.3
gamma_min = best_gamma * 0.7
gamma_max = best_gamma * 1.3

param_rbf_refined = {
    'svm__kernel': ['rbf'],
    'svm__C': np.linspace(C_min, C_max, 20),
    'svm__gamma': np.linspace(gamma_min, gamma_max, 20)
}

print(f"Recherche autour de C={best_C:.4f}, gamma={best_gamma:.6f}")
print(f"Plage C: [{C_min:.4f}, {C_max:.4f}]")
print(f"Plage gamma: [{gamma_min:.6f}, {gamma_max:.6f}]")
print(f"Nombre de combinaisons: {len(param_rbf_refined['svm__C']) * len(param_rbf_refined['svm__gamma'])}")
print("\nRecherche en cours...\n")

grid_rbf_refined = GridSearchCV(pipe, param_rbf_refined, cv=kf, scoring='accuracy', n_jobs=-1)
grid_rbf_refined.fit(X_train, y_train)

# print("\n" + "-"*70)
# print("RÉSULTATS RECHERCHE RAFFINÉE:")
# print("-"*70)
# print(f"Meilleurs paramètres: {grid_rbf_refined.best_params_}")
# print(f"Meilleure accuracy (CV): {grid_rbf_refined.best_score_:.6f}")

# Test sur l'ensemble de test
y_pred_refined = grid_rbf_refined.predict(X_test)
test_accuracy_refined = np.mean(y_pred_refined == y_test)
print(f"Accuracy sur le test: {test_accuracy_refined:.6f}")

# Top 10 pour la recherche raffinée
results_refined = pd.DataFrame(grid_rbf_refined.cv_results_)
top_10_refined = results_refined.nlargest(10, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
print("\nTop 10 des meilleures combinaisons raffinées:")
print("-"*70)
for idx, row in top_10_refined.iterrows():
    C_val = row['params']['svm__C']
    gamma_val = row['params']['svm__gamma']
    print(f"Score CV: {row['mean_test_score']:.6f} (±{row['std_test_score']:.6f})")
    print(f"  C={C_val:.4f}, gamma={gamma_val:.6f}")

# ========== RÉSUMÉ FINAL ==========
print("\n\n" + "="*70)
print("RÉSUMÉ FINAL")
print("="*70)

# Choisir le meilleur entre les deux recherches
if grid_rbf_refined.best_score_ > grid_rbf_logspace.best_score_:
    final_best = grid_rbf_refined.best_params_
    final_score = grid_rbf_refined.best_score_
    final_test = test_accuracy_refined
    source = "recherche raffinée"
else:
    final_best = grid_rbf_logspace.best_params_
    final_score = grid_rbf_logspace.best_score_
    final_test = test_accuracy
    source = "logspace"

print(f"\nMeilleur résultat de la {source}:")
print(f"  CV accuracy: {final_score:.6f}")
print(f"  Test accuracy: {final_test:.6f}")
print(f"  C: {final_best['svm__C']:.6f}")
print(f"  gamma: {final_best['svm__gamma']:.8f}")

print("\n>>> Code pour la soumission <<<")
print(f"my_rbf_svm = SVC(kernel='rbf', C={final_best['svm__C']:.6f}, gamma={final_best['svm__gamma']:.8f})")

print("\n" + "="*70)

# Comparaison avec l'objectif
target_cv = 0.8727
target_test = 0.8731

print("\n>>> COMPARAISON AVEC L'OBJECTIF <<<")
print(f"Objectif CV:   {target_cv:.6f}  |  Obtenu: {final_score:.6f}  |  Diff: {(final_score - target_cv):.6f}")
print(f"Objectif Test: {target_test:.6f}  |  Obtenu: {final_test:.6f}  |  Diff: {(final_test - target_test):.6f}")

if final_score >= target_cv:
    print("\n✓ Objectif CV atteint ou dépassé!")
else:
    print(f"\n✗ Objectif CV manqué de {(target_cv - final_score):.6f}")

if final_test >= target_test:
    print("✓ Objectif Test atteint ou dépassé!")
else:
    print(f"✗ Objectif Test manqué de {(target_test - final_test):.6f}")

print("\n" + "="*70)

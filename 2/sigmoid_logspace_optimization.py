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

print("="*70)
print("OPTIMISATION SIGMOID AVEC LOGSPACE")
print("="*70)
print(f"Recherche des meilleurs paramètres pour le noyau sigmoid")
print("="*70)

# ========== RECHERCHE FINE AVEC LOGSPACE ==========
print("\n" + "="*70)
print("1. RECHERCHE FINE AVEC LOGSPACE")
print("="*70)

# Utilisation de logspace pour explorer les paramètres
# C: de 10^0 à 10^3 (1 à 1000)
# gamma: de 10^-4 à 10^0 (0.0001 à 1) + 'scale' et 'auto'
# coef0: de -5 à 5
param_sigmoid_logspace = {
    'svm__kernel': ['sigmoid'],
    'svm__C': np.logspace(0, 2, 30),  # 30 valeurs entre 1 et 100
    'svm__gamma': ['scale', 'auto'] + list(np.logspace(-3, 0, 30)),  # 'scale', 'auto' + 30 valeurs entre 0.001 et 1
    'svm__coef0': np.linspace(-5, 5, 10)  # 20 valeurs entre -5 et 5
}

print(f"Nombre de combinaisons: {len(param_sigmoid_logspace['svm__C']) * len(param_sigmoid_logspace['svm__gamma']) * len(param_sigmoid_logspace['svm__coef0'])}")
print("Plage C:", f"[{param_sigmoid_logspace['svm__C'][0]:.4f}, {param_sigmoid_logspace['svm__C'][-1]:.4f}]")
print("Plage gamma:", f"['scale', 'auto', {param_sigmoid_logspace['svm__gamma'][2]:.6f}, ..., {param_sigmoid_logspace['svm__gamma'][-1]:.6f}]")
print("Plage coef0:", f"[{param_sigmoid_logspace['svm__coef0'][0]:.2f}, {param_sigmoid_logspace['svm__coef0'][-1]:.2f}]")
print("\nRecherche en cours...\n")

grid_sigmoid_logspace = GridSearchCV(pipe, param_sigmoid_logspace, cv=kf, scoring='accuracy', n_jobs=-1)
grid_sigmoid_logspace.fit(X_train, y_train)

print("\n" + "-"*70)
print("RÉSULTATS LOGSPACE:")
print("-"*70)
print(f"Meilleurs paramètres: {grid_sigmoid_logspace.best_params_}")
print(f"Meilleure accuracy (CV): {grid_sigmoid_logspace.best_score_:.6f}")

# Test sur l'ensemble de test
y_pred = grid_sigmoid_logspace.predict(X_test)
test_accuracy = np.mean(y_pred == y_test)
print(f"Accuracy sur le test: {test_accuracy:.6f}")

# Top 10 pour Sigmoid
results_sigmoid = pd.DataFrame(grid_sigmoid_logspace.cv_results_)
top_10_sigmoid = results_sigmoid.nlargest(10, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
print("\nTop 10 des meilleures combinaisons:")
print("-"*70)
for idx, row in top_10_sigmoid.iterrows():
    C_val = row['params']['svm__C']
    gamma_val = row['params']['svm__gamma']
    coef0_val = row['params']['svm__coef0']
    print(f"Score CV: {row['mean_test_score']:.6f} (±{row['std_test_score']:.6f})")
    if isinstance(gamma_val, str):
        print(f"  C={C_val:.4f}, gamma='{gamma_val}', coef0={coef0_val:.4f}")
    else:
        print(f"  C={C_val:.4f}, gamma={gamma_val:.6f}, coef0={coef0_val:.4f}")

best_sigmoid = grid_sigmoid_logspace.best_params_

# ========== RECHERCHE RAFFINÉE AUTOUR DU MEILLEUR ==========
print("\n\n" + "="*70)
print("2. RECHERCHE RAFFINÉE AUTOUR DU MEILLEUR")
print("="*70)

# Créer une grille raffinée autour des meilleurs paramètres
best_C = best_sigmoid['svm__C']
best_gamma = best_sigmoid['svm__gamma']
best_coef0 = best_sigmoid['svm__coef0']

# Si gamma est 'scale' ou 'auto', on ne fait pas de recherche raffinée sur gamma
if isinstance(best_gamma, str):
    print(f"Gamma optimal est '{best_gamma}', pas de recherche raffinée sur gamma")
    gamma_refined = [best_gamma]
else:
    # Recherche dans un voisinage de ±30% autour des meilleurs paramètres
    gamma_min = best_gamma * 0.7
    gamma_max = best_gamma * 1.3
    gamma_refined = list(np.linspace(gamma_min, gamma_max, 15))

# Recherche dans un voisinage de ±30% autour des meilleurs paramètres
C_min = best_C * 0.7
C_max = best_C * 1.3
coef0_min = best_coef0 - 1.0
coef0_max = best_coef0 + 1.0

param_sigmoid_refined = {
    'svm__kernel': ['sigmoid'],
    'svm__C': np.linspace(C_min, C_max, 15),
    'svm__gamma': gamma_refined,
    'svm__coef0': np.linspace(coef0_min, coef0_max, 15)
}

print(f"Recherche autour de C={best_C:.4f}, gamma={best_gamma}, coef0={best_coef0:.4f}")
print(f"Plage C: [{C_min:.4f}, {C_max:.4f}]")
if isinstance(best_gamma, str):
    print(f"Gamma: '{best_gamma}'")
else:
    print(f"Plage gamma: [{gamma_min:.6f}, {gamma_max:.6f}]")
print(f"Plage coef0: [{coef0_min:.4f}, {coef0_max:.4f}]")
print(f"Nombre de combinaisons: {len(param_sigmoid_refined['svm__C']) * len(param_sigmoid_refined['svm__gamma']) * len(param_sigmoid_refined['svm__coef0'])}")
print("\nRecherche en cours...\n")

grid_sigmoid_refined = GridSearchCV(pipe, param_sigmoid_refined, cv=kf, scoring='accuracy', n_jobs=-1)
grid_sigmoid_refined.fit(X_train, y_train)

print("\n" + "-"*70)
print("RÉSULTATS RECHERCHE RAFFINÉE:")
print("-"*70)
print(f"Meilleurs paramètres: {grid_sigmoid_refined.best_params_}")
print(f"Meilleure accuracy (CV): {grid_sigmoid_refined.best_score_:.6f}")

# Test sur l'ensemble de test
y_pred_refined = grid_sigmoid_refined.predict(X_test)
test_accuracy_refined = np.mean(y_pred_refined == y_test)
print(f"Accuracy sur le test: {test_accuracy_refined:.6f}")

# Top 10 pour la recherche raffinée
results_refined = pd.DataFrame(grid_sigmoid_refined.cv_results_)
top_10_refined = results_refined.nlargest(10, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
print("\nTop 10 des meilleures combinaisons raffinées:")
print("-"*70)
for idx, row in top_10_refined.iterrows():
    C_val = row['params']['svm__C']
    gamma_val = row['params']['svm__gamma']
    coef0_val = row['params']['svm__coef0']
    print(f"Score CV: {row['mean_test_score']:.6f} (±{row['std_test_score']:.6f})")
    if isinstance(gamma_val, str):
        print(f"  C={C_val:.4f}, gamma='{gamma_val}', coef0={coef0_val:.4f}")
    else:
        print(f"  C={C_val:.4f}, gamma={gamma_val:.6f}, coef0={coef0_val:.4f}")

# ========== RÉSUMÉ FINAL ==========
print("\n\n" + "="*70)
print("RÉSUMÉ FINAL")
print("="*70)

# Choisir le meilleur entre les deux recherches
if grid_sigmoid_refined.best_score_ > grid_sigmoid_logspace.best_score_:
    final_best = grid_sigmoid_refined.best_params_
    final_score = grid_sigmoid_refined.best_score_
    final_test = test_accuracy_refined
    source = "recherche raffinée"
else:
    final_best = grid_sigmoid_logspace.best_params_
    final_score = grid_sigmoid_logspace.best_score_
    final_test = test_accuracy
    source = "logspace"

print(f"\nMeilleur résultat de la {source}:")
print(f"  CV accuracy: {final_score:.6f}")
print(f"  Test accuracy: {final_test:.6f}")
print(f"  C: {final_best['svm__C']:.6f}")
if isinstance(final_best['svm__gamma'], str):
    print(f"  gamma: '{final_best['svm__gamma']}'")
else:
    print(f"  gamma: {final_best['svm__gamma']:.8f}")
print(f"  coef0: {final_best['svm__coef0']:.6f}")

print("\n>>> Code pour la soumission <<<")
if isinstance(final_best['svm__gamma'], str):
    print(f"my_sigm_svm = SVC(kernel='sigmoid', C={final_best['svm__C']:.6f}, gamma='{final_best['svm__gamma']}', coef0={final_best['svm__coef0']:.6f})")
else:
    print(f"my_sigm_svm = SVC(kernel='sigmoid', C={final_best['svm__C']:.6f}, gamma={final_best['svm__gamma']:.8f}, coef0={final_best['svm__coef0']:.6f})")

print("\n" + "="*70)

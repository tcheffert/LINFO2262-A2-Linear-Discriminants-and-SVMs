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

#------- Current best -------#
# my_poly_svm = SVC(kernel='poly', degree=2, C=0.003, gamma=1.5, coef0=3.75)  #0.868182
#----------------------------#

# Paramètres pour le noyau Polynomial (Question 3)
# Recherche élargie pour trouver les meilleurs paramètres
param_poly = {
    'svm__kernel': ['poly'],
    'svm__degree': [2, 3, 4],  
    'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  
    'svm__gamma': ['scale', 'auto', 1, 0.5, 0.1, 0.01, 0.001],  
    'svm__coef0': [0, 0.1, 0.5, 1, 2, 3, 4, 5] 
}

print("=== Recherche des meilleurs paramètres pour le noyau polynomial ===")
print(f"Nombre total de combinaisons: {np.prod([len(v) for v in param_poly.values()])}")
print("Cela peut prendre quelques minutes...\n")

# Lance la recherche exhaustive
grid = GridSearchCV(pipe, param_poly, cv=kf, scoring='accuracy', verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)

# Affichage des résultats
print("\n" + "="*70)
print(f"Meilleurs paramètres trouvés: {grid.best_params_}")
print(f"Meilleure accuracy (CV): {grid.best_score_:.6f}")
print("="*70)

# Affichage du top 10 des meilleures combinaisons
results_df = pd.DataFrame(grid.cv_results_)
top_10 = results_df.nlargest(10, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
print("\nTop 10 des meilleures combinaisons:")
print("-"*70)
for idx, row in top_10.iterrows():
    print(f"Score: {row['mean_test_score']:.6f} (±{row['std_test_score']:.6f})")
    print(f"  Params: {row['params']}")
    print()

# Extraction des meilleurs paramètres
best_params = grid.best_params_
print("\n" + "="*70)
print("SOLUTION FINALE POUR LA SOUMISSION:")
print("="*70)
print(f"degree={best_params['svm__degree']}")
print(f"C={best_params['svm__C']}")
print(f"gamma='{best_params['svm__gamma']}'" if isinstance(best_params['svm__gamma'], str) else f"gamma={best_params['svm__gamma']}")
print(f"coef0={best_params['svm__coef0']}")
print("="*70)
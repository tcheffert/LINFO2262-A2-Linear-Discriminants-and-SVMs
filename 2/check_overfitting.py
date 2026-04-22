import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

train_df = pd.read_csv("../HeartTrain.csv")
test_df = pd.read_csv("../HeartTest.csv")

X_train = train_df.drop(columns=['labels'])
y_train = train_df['labels']
X_test = test_df.drop(columns=['labels'])
y_test = test_df['labels']

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kf = KFold(n_splits=10, shuffle=False)

def evaluate_config(name, svm_model):
    """Évalue un modèle sur CV et Test"""
    # CV Score
    cv_scores = cross_val_score(svm_model, X_train_scaled, y_train, cv=kf, scoring='accuracy')
    cv_score = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # Test Score
    svm_model.fit(X_train_scaled, y_train)
    test_score = svm_model.score(X_test_scaled, y_test)
    
    gap = cv_score - test_score
    
    return {
        'name': name,
        'cv_score': cv_score,
        'cv_std': cv_std,
        'test_score': test_score,
        'gap': gap
    }

print("="*80)
print("DÉTECTION D'OVERFITTING - COMPARAISON CV vs TEST")
print("="*80)
print("\nUn gap important (CV >> Test) indique de l'overfitting\n")

# ========== POLYNOMIAL ==========
print("="*80)
print("KERNEL POLYNOMIAL")
print("="*80)

poly_configs = [
    ("Tes params actuels", {'degree': 2, 'C': 0.003, 'gamma': 1.5, 'coef0': 3.75}),
    ("C plus grand", {'degree': 2, 'C': 0.01, 'gamma': 1, 'coef0': 3}),
    ("Gamma scale", {'degree': 2, 'C': 0.01, 'gamma': 'scale', 'coef0': 1}),
    ("Plus simple", {'degree': 2, 'C': 1, 'gamma': 'scale', 'coef0': 1}),
    ("Très régularisé", {'degree': 2, 'C': 0.1, 'gamma': 'scale', 'coef0': 0.5}),
]

poly_results = []
for name, params in poly_configs:
    svm = SVC(kernel='poly', **params)
    result = evaluate_config(name, svm)
    poly_results.append(result)
    print(f"{name:25s} | CV={result['cv_score']:.4f}±{result['cv_std']:.3f} Test={result['test_score']:.4f} Gap={result['gap']:+.4f}")

# ========== RBF ==========
print("\n" + "="*80)
print("KERNEL RBF")
print("="*80)

rbf_configs = [
    ("Tes params actuels", {'C': 25, 'gamma': 'scale'}),
    ("C=5 (trouvé mieux)", {'C': 5, 'gamma': 'scale'}),
    ("C=10", {'C': 10, 'gamma': 'scale'}),
    ("C=15", {'C': 15, 'gamma': 'scale'}),
    ("C=1 très régularisé", {'C': 1, 'gamma': 'scale'}),
    ("Gamma numérique", {'C': 10, 'gamma': 0.05}),
]

rbf_results = []
for name, params in rbf_configs:
    svm = SVC(kernel='rbf', **params)
    result = evaluate_config(name, svm)
    rbf_results.append(result)
    print(f"{name:25s} | CV={result['cv_score']:.4f}±{result['cv_std']:.3f} Test={result['test_score']:.4f} Gap={result['gap']:+.4f}")

# ========== SIGMOID ==========
print("\n" + "="*80)
print("KERNEL SIGMOID")
print("="*80)

sigmoid_configs = [
    ("Tes params actuels", {'C': 75, 'gamma': 0.01, 'coef0': -1}),
    ("C plus petit", {'C': 50, 'gamma': 0.01, 'coef0': -1}),
    ("C plus grand", {'C': 100, 'gamma': 0.01, 'coef0': -1}),
    ("Gamma différent", {'C': 75, 'gamma': 0.05, 'coef0': 0}),
    ("Plus simple", {'C': 10, 'gamma': 'scale', 'coef0': 0}),
]

sigmoid_results = []
for name, params in sigmoid_configs:
    svm = SVC(kernel='sigmoid', **params)
    result = evaluate_config(name, svm)
    sigmoid_results.append(result)
    print(f"{name:25s} | CV={result['cv_score']:.4f}±{result['cv_std']:.3f} Test={result['test_score']:.4f} Gap={result['gap']:+.4f}")

# ========== ANALYSE GLOBALE ==========
print("\n" + "="*80)
print("ANALYSE: MEILLEUR PAR TEST ACCURACY (le vrai critère!)")
print("="*80)

all_results = []
all_results.extend([(r, 'POLY') for r in poly_results])
all_results.extend([(r, 'RBF') for r in rbf_results])
all_results.extend([(r, 'SIGMOID') for r in sigmoid_results])

# Trier par test accuracy
all_results.sort(key=lambda x: x[0]['test_score'], reverse=True)

print("\nTop 10 configurations par Test Accuracy:")
print("-"*80)
for i, (result, kernel) in enumerate(all_results[:10], 1):
    print(f"\n{i}. [{kernel}] {result['name']}")
    print(f"   Test: {result['test_score']:.4f} | CV: {result['cv_score']:.4f} | Gap: {result['gap']:+.4f}")

# Analyse du gap
print("\n" + "="*80)
print("ANALYSE DU GAP (overfitting)")
print("="*80)
print("Gap < 0.01: Excellent (bonne généralisation)")
print("Gap 0.01-0.02: Bon")
print("Gap 0.02-0.04: Acceptable")
print("Gap > 0.04: Attention, possible overfitting!")
print("-"*80)

for kernel_name, results in [('POLYNOMIAL', poly_results), ('RBF', rbf_results), ('SIGMOID', sigmoid_results)]:
    print(f"\n{kernel_name}:")
    best_test = max(results, key=lambda x: x['test_score'])
    current = results[0]  # Tes params actuels
    
    print(f"  Tes params actuels:")
    print(f"    Test={current['test_score']:.4f}, Gap={current['gap']:+.4f}")
    
    if current != best_test:
        print(f"  ⚠️  Meilleur trouvé: {best_test['name']}")
        print(f"    Test={best_test['test_score']:.4f}, Gap={best_test['gap']:+.4f}")
        print(f"    Amélioration: +{(best_test['test_score'] - current['test_score']):.4f}")
    else:
        print(f"  ✅ Tes params sont les meilleurs!")

print("\n" + "="*80)
print("RECOMMANDATIONS FINALES")
print("="*80)

# Trouver les meilleurs de chaque kernel
best_poly = max(poly_results, key=lambda x: x['test_score'])
best_rbf = max(rbf_results, key=lambda x: x['test_score'])
best_sigmoid = max(sigmoid_results, key=lambda x: x['test_score'])

print(f"\nPOLYNOMIAL: {best_poly['name']}")
print(f"  Test: {best_poly['test_score']:.4f}, Gap: {best_poly['gap']:+.4f}")

print(f"\nRBF: {best_rbf['name']}")
print(f"  Test: {best_rbf['test_score']:.4f}, Gap: {best_rbf['gap']:+.4f}")

print(f"\nSIGMOID: {best_sigmoid['name']}")
print(f"  Test: {best_sigmoid['test_score']:.4f}, Gap: {best_sigmoid['gap']:+.4f}")

overall_best = max([best_poly, best_rbf, best_sigmoid], key=lambda x: x['test_score'])
print(f"\n🏆 MEILLEUR GLOBAL: {overall_best['name']}")
print(f"   Test Accuracy: {overall_best['test_score']:.4f}")

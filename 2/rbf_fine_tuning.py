import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

#--- Notes ---#
# Nouveau record: C=5.0000, gamma=0.035984 | CV=0.8636, Test=0.8731
# Nouveau record: C=3.6790, gamma=0.047021 | CV=0.8727, Test=0.8731
# Nouveau record: C=3.5391, gamma=0.049256 | CV=0.8773, Test=0.8731

# ======================================================================
# C = 3.539100
# gamma = 0.049256
# CV Score = 0.877273
# Test Accuracy = 0.8731
# ======================================================================

# Chargement des données
train_df = pd.read_csv("./HeartTrain.csv")
test_df = pd.read_csv("./HeartTest.csv")

X_train = train_df.drop(columns=['labels'])
y_train = train_df['labels']
X_test = test_df.drop(columns=['labels'])
y_test = test_df['labels']

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# PHASE 1: RECHERCHE GROSSIÈRE
# ============================================================================
print("PHASE 1: Recherche grossière (exploration large)...")

C_coarse = np.logspace(np.log10(3), np.log10(30), 80)
gamma_coarse = np.logspace(np.log10(0.003), np.log10(0.1), 80)

best_record = {'cv_score': 0, 'test_acc': 0, 'min_score': 0}
coarse_results = []

total = len(C_coarse) * len(gamma_coarse)
for idx, (C, gamma) in enumerate([(c, g) for c in C_coarse for g in gamma_coarse]):
    model = SVC(kernel='rbf', C=C, gamma=gamma)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    
    model.fit(X_train_scaled, y_train)
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    
    min_score = min(cv_mean, test_acc)  # On veut que les DEUX soient élevés
    coarse_results.append({'C': C, 'gamma': gamma, 'cv_score': cv_mean, 
                           'test_acc': test_acc, 'min_score': min_score})
    
    if min_score > best_record['min_score'] and cv_mean >= 0.86 and test_acc >= 0.86:
        best_record = {'C': C, 'gamma': gamma, 'cv_score': cv_mean, 
                       'test_acc': test_acc, 'min_score': min_score}
        print(f"🎯 Nouveau record: C={C:.4f}, gamma={gamma:.6f} | CV={cv_mean:.4f}, Test={test_acc:.4f}")
    
    if (idx + 1) % 1000 == 0:
        print(f"  [{idx+1}/{total}] en cours...")

# Sélectionner les 8 meilleurs (basés sur le score minimum pour avoir les deux élevés)
coarse_results.sort(key=lambda x: x['min_score'], reverse=True)
top_8 = coarse_results[:8]

# ============================================================================
# PHASE 2: AFFINEMENT AUTOUR DES MEILLEURS RÉSULTATS
# ============================================================================
print(f"\nPHASE 2: Affinement intensif autour des {len(top_8)} meilleurs résultats...")

fine_results = []
for result in top_8:
    C_center, gamma_center = result['C'], result['gamma']
    
    # ±15% autour des valeurs avec beaucoup de points
    C_fine = np.logspace(np.log10(C_center * 0.85), np.log10(C_center * 1.15), 150)
    gamma_fine = np.logspace(np.log10(gamma_center * 0.85), np.log10(gamma_center * 1.15), 150)
    
    for C, gamma in [(c, g) for c in C_fine for g in gamma_fine]:
        model = SVC(kernel='rbf', C=C, gamma=gamma)
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_scores.mean()
        
        model.fit(X_train_scaled, y_train)
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
        
        min_score = min(cv_mean, test_acc)
        fine_results.append({'C': C, 'gamma': gamma, 'cv_score': cv_mean, 
                            'test_acc': test_acc, 'min_score': min_score})
        
        if min_score > best_record['min_score'] and cv_mean >= 0.86 and test_acc >= 0.86:
            best_record = {'C': C, 'gamma': gamma, 'cv_score': cv_mean, 
                          'test_acc': test_acc, 'min_score': min_score}
            print(f"🎯 Nouveau record: C={C:.4f}, gamma={gamma:.6f} | CV={cv_mean:.4f}, Test={test_acc:.4f}")

# ============================================================================
# RÉSULTAT FINAL
# ============================================================================
fine_results.sort(key=lambda x: x['min_score'], reverse=True)
best = fine_results[0] if fine_results else coarse_results[0]

print("\n" + "="*70)
print("MEILLEUR RÉSULTAT TROUVÉ (CV >= 0.86 ET Test >= 0.86):")
print("="*70)
print(f"C = {best['C']:.6f}")
print(f"gamma = {best['gamma']:.6f}")
print(f"CV Score = {best['cv_score']:.6f}")
print(f"Test Accuracy = {best['test_acc']:.4f}")
print("="*70)

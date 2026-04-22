import pandas as pd
import numpy as np
import sys
sys.path.append('/home/theo/Documents/UCL/Q10/ML/A2/3')
from q1 import perceptron

# Chargement des données
train_df = pd.read_csv("/home/theo/Documents/UCL/Q10/ML/A2/HeartTrain.csv")
test_df = pd.read_csv("/home/theo/Documents/UCL/Q10/ML/A2/HeartTest.csv")

print("="*70)
print("TEST DE LA FONCTION PERCEPTRON")
print("="*70)

# Préparation des données d'entraînement
X_train = train_df.drop(columns=['labels'])
y_train = train_df['labels'].to_numpy()

X_test = test_df.drop(columns=['labels'])
y_test = test_df['labels'].to_numpy()

print(f"\nDimensions:")
print(f"  X_train: {X_train.shape}")
print(f"  y_train: {y_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  y_test: {y_test.shape}")

# Paramètres du perceptron
d = X_train.shape[1]  # nombre de features
w_init = np.zeros(d + 1)  # d+1 car on a le biais (intercept)
b = 1.0  # marge
eta = 0.1  # learning rate
max_epochs = 100

print(f"\nParamètres:")
print(f"  Nombre de features (d): {d}")
print(f"  Marge (b): {b}")
print(f"  Learning rate (eta): {eta}")
print(f"  Max epochs: {max_epochs}")

# Test 1: Sans decay
print("\n" + "-"*70)
print("TEST 1: Perceptron SANS decay")
print("-"*70)
w1 = perceptron(X_train, y_train, b, w_init, eta, max_epochs, decay=False)
print(f"Poids obtenus: {w1[:5]}... (premiers 5 éléments)")

# Fonction pour faire des prédictions
def predict(X, w):
    """Fait des prédictions avec le perceptron"""
    n_samples = X.shape[0]
    X_aug = np.hstack([np.ones((n_samples, 1)), X.to_numpy()])
    scores = np.dot(X_aug, w)
    # Conversion -1/1 vers 0/1
    predictions = (scores > 0).astype(int)
    return predictions

# Prédictions et accuracy
y_train_pred = predict(X_train, w1)
y_test_pred = predict(X_test, w1)

train_acc = np.mean(y_train_pred == y_train)
test_acc = np.mean(y_test_pred == y_test)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Test 2: Avec decay
print("\n" + "-"*70)
print("TEST 2: Perceptron AVEC decay (eta/k)")
print("-"*70)
w2 = perceptron(X_train, y_train, b, w_init, eta, max_epochs, decay=True)
print(f"Poids obtenus: {w2[:5]}... (premiers 5 éléments)")

y_train_pred2 = predict(X_train, w2)
y_test_pred2 = predict(X_test, w2)

train_acc2 = np.mean(y_train_pred2 == y_train)
test_acc2 = np.mean(y_test_pred2 == y_test)

print(f"Train Accuracy: {train_acc2:.4f}")
print(f"Test Accuracy: {test_acc2:.4f}")

# Test 3: Différentes valeurs de marge
print("\n" + "-"*70)
print("TEST 3: Influence de la marge (b)")
print("-"*70)
margins = [0.5, 1.0, 2.0, 5.0]
for b_test in margins:
    w_test = perceptron(X_train, y_train, b_test, w_init, eta, max_epochs, decay=False)
    y_pred_test = predict(X_test, w_test)
    acc_test = np.mean(y_pred_test == y_test)
    print(f"  b={b_test:4.1f} -> Test Accuracy: {acc_test:.4f}")

# Test 4: Différents learning rates
print("\n" + "-"*70)
print("TEST 4: Influence du learning rate (eta)")
print("-"*70)
learning_rates = [0.01, 0.1, 0.5, 1.0]
for eta_test in learning_rates:
    w_test = perceptron(X_train, y_train, b, w_init, eta_test, max_epochs, decay=False)
    y_pred_test = predict(X_test, w_test)
    acc_test = np.mean(y_pred_test == y_test)
    print(f"  eta={eta_test:5.2f} -> Test Accuracy: {acc_test:.4f}")

print("\n" + "="*70)
print("TESTS TERMINÉS")
print("="*70)

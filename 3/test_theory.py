import pandas as pd
import numpy as np
from q1 import perceptron
from q3 import lms
from q2 import predict

# Charger les données
train = pd.read_csv('../HeartTrain.csv')
test = pd.read_csv('../HeartTest.csv')

X_train = train.drop('labels', axis=1)
y_train = train['labels'].values
X_test = test.drop('labels', axis=1)
y_test = test['labels'].values

# Normalisation
mean = X_train.mean()
std = X_train.std()
X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

d = X_train.shape[1]
b = 0.0
eta = 0.01
epoch = 10000

print("=== Test 1: Influence de w_init sur Perceptron ===")
test_accs_decay = []
test_accs_no_decay = []
for seed in [0, 42, 123, 456, 789]:
    np.random.seed(seed)
    w_init = np.random.randn(d + 1) * 0.01
    
    # Avec decay
    w = perceptron(X_train_norm, y_train, b, w_init, eta, epoch, decay=True)
    pred = predict(w, X_test_norm)
    test_accs_decay.append(np.mean(pred == y_test))
    
    # Sans decay
    w = perceptron(X_train_norm, y_train, b, w_init, eta, epoch, decay=False)
    pred = predict(w, X_test_norm)
    test_accs_no_decay.append(np.mean(pred == y_test))

print(f"Avec decay - Test acc: {test_accs_decay}")
print(f"Variation: {np.std(test_accs_decay):.4f}")
print(f"Sans decay - Test acc: {test_accs_no_decay}")
print(f"Variation: {np.std(test_accs_no_decay):.4f}")

print("\n=== Test 2: Convergence LMS sans decay ===")
w_init = np.zeros(d + 1)
for eta_test in [0.001, 0.01, 0.1]:
    w = lms(X_train_norm, y_train, b=1.0, w_init=w_init, eta=eta_test, epoch=100, decay=False)
    pred_train = predict(w, X_train_norm)
    pred_test = predict(w, X_test_norm)
    print(f"eta={eta_test} (no decay): Train={np.mean(pred_train == y_train):.4f}, Test={np.mean(pred_test == y_test):.4f}")

print("\n=== Test 3: Perceptron avec epochs élevés (convergence?) ===")
w_init = np.zeros(d + 1)
for ep in [100, 1000, 10000, 100000]:
    w = perceptron(X_train_norm, y_train, b=0.0, w_init=w_init, eta=0.01, epoch=ep, decay=True)
    pred_train = predict(w, X_train_norm)
    print(f"Epoch={ep}: Train accuracy = {np.mean(pred_train == y_train):.4f}")

print("\n=== Test 4: Comparaison Perceptron vs LMS (mêmes paramètres) ===")
w_init = np.zeros(d + 1)
b = 1.0
eta = 0.01
epoch = 100

w_percep = perceptron(X_train_norm, y_train, b, w_init, eta, epoch, decay=True)
w_lms = lms(X_train_norm, y_train, b, w_init, eta, epoch, decay=True)

pred_percep = predict(w_percep, X_train_norm)
pred_lms = predict(w_lms, X_train_norm)

print(f"Perceptron train acc: {np.mean(pred_percep == y_train):.4f}")
print(f"LMS train acc: {np.mean(pred_lms == y_train):.4f}")
print(f"Même solution? {np.allclose(w_percep, w_lms)}")

print("\n=== Test 5: Influence du ratio b/eta sur convergence ===")
w_init = np.zeros(d + 1)
epoch = 1000

configs = [
    (0.0, 0.01),   # b/eta = 0
    (0.5, 0.01),   # b/eta = 50
    (1.0, 0.01),   # b/eta = 100
    (1.0, 0.001),  # b/eta = 1000
]

for b_test, eta_test in configs:
    w = perceptron(X_train_norm, y_train, b_test, w_init, eta_test, epoch, decay=True)
    pred_train = predict(w, X_train_norm)
    print(f"b={b_test}, eta={eta_test}, ratio={b_test/eta_test if eta_test > 0 else 0:.0f}: Train={np.mean(pred_train == y_train):.4f}")

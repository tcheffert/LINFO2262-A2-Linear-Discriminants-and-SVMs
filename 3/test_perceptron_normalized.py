import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('/home/theo/Documents/UCL/Q10/ML/A2/3')
from q1 import perceptron

# Chargement des données
train_df = pd.read_csv("/home/theo/Documents/UCL/Q10/ML/A2/HeartTrain.csv")
test_df = pd.read_csv("/home/theo/Documents/UCL/Q10/ML/A2/HeartTest.csv")

print("="*70)
print("TEST AVEC NORMALISATION DES FEATURES")
print("="*70)

# Préparation avec StandardScaler
X_train = train_df.drop(columns=['labels'])
y_train = train_df['labels'].to_numpy()
X_test = test_df.drop(columns=['labels'])
y_test = test_df['labels'].to_numpy()

# Normalisation
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns
)

# Paramètres
d = X_train_scaled.shape[1]
w_init = np.zeros(d + 1)
b = 1.0
eta = 0.1
max_epochs = 100

def predict(X, w):
    n_samples = X.shape[0]
    X_aug = np.hstack([np.ones((n_samples, 1)), X.values])
    scores = np.dot(X_aug, w)
    return (scores > 0).astype(int)

print("\nTest AVEC normalisation + decay:")
w = perceptron(X_train_scaled, y_train, b, w_init, eta, max_epochs, decay=True)
y_train_pred = predict(X_train_scaled, w)
y_test_pred = predict(X_test_scaled, w)

train_acc = np.mean(y_train_pred == y_train)
test_acc = np.mean(y_test_pred == y_test)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Poids (premiers 5): {w[:5]}")

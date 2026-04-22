import pandas as pd
import numpy as np
from q3 import lms
from q2 import predict

# Charger les données
train = pd.read_csv('./HeartTrain.csv')
test = pd.read_csv('./HeartTest.csv')

X_train = train.drop('labels', axis=1)
y_train = train['labels'].values

X_test = test.drop('labels', axis=1)
y_test = test['labels'].values

# Normalisation (comme pour le perceptron)
mean = X_train.mean()
std = X_train.std()
X_train_norm = (X_train - mean) / std
X_test_norm = (X_test - mean) / std

# Paramètres
d = X_train.shape[1]
w_init = np.zeros(d + 1)
b = 1.0
eta = 0.01
epoch = 100

print("=== Test LMS avec decay ===")
w_lms_decay = lms(X_train_norm, y_train, b, w_init, eta, epoch, decay=True)
predictions_train = predict(w_lms_decay, X_train_norm)
predictions_test = predict(w_lms_decay, X_test_norm)

train_acc = np.mean(predictions_train == y_train)
test_acc = np.mean(predictions_test == y_test)

print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

print("\n=== Test LMS sans decay ===")
w_lms_no_decay = lms(X_train_norm, y_train, b, w_init, eta, epoch, decay=False)
predictions_train = predict(w_lms_no_decay, X_train_norm)
predictions_test = predict(w_lms_no_decay, X_test_norm)

train_acc = np.mean(predictions_train == y_train)
test_acc = np.mean(predictions_test == y_test)

print(f"Train accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

print("\n=== Test avec différents eta ===")
for eta_test in [0.001, 0.01, 0.1, 0.5]:
    w_lms = lms(X_train_norm, y_train, b, w_init, eta_test, epoch, decay=True)
    predictions_test = predict(w_lms, X_test_norm)
    test_acc = np.mean(predictions_test == y_test)
    print(f"eta={eta_test}: Test accuracy = {test_acc:.4f}")

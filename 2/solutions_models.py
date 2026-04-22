from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Linéaire : 0.8325

# Polynomial kernel parameters
my_poly_svm = SVC(kernel='poly', degree=2, C=0.003, gamma=1.5, coef0=3.75)  #0.868182 -> 0.8426

# RBF kernel parameters
my_rbf_svm = SVC(kernel='rbf', C=10.826367, gamma=0.01610262) #0.872727 -> 0.8376
# my_rbf_svm = SVC(kernel='rbf', C=5, gamma='scale') #0.8636 -> 0.8528
# my_rbf_svm = SVC(kernel='rbf', C=13.538762, gamma=0.014636) #0.872727 -> 0.8528

# Sigmoid kernel parameters
my_sigm_svm = SVC(kernel='sigmoid', C=75, gamma=0.01, coef0=-1) #0.859091 -> 0.8528

#==== Q5 ====#

train_df = pd.read_csv("./HeartTrain.csv")
test_df = pd.read_csv("./HeartTest.csv")


# Préparation des données
X_train = train_df.drop(columns=['labels'])
y_train = train_df['labels']
X_test = test_df.drop(columns=['labels'])
y_test = test_df['labels']

# On scale sur le train et on applique sur le test
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Tes 4 modèles
models = {
    "linear": SVC(kernel='linear'),
    "poly": SVC(kernel='poly', degree=2, C=0.003, gamma=1.5, coef0=3.75),
    "rbf": SVC(kernel='rbf', C=3.539100, gamma=0.049256),
    "sigmoid": SVC(kernel='sigmoid', C=41.871879, gamma='scale', coef0=-1.523810)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"Modèle {name}: Test Accuracy = {acc:.4f}")
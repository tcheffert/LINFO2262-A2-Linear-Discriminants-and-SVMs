import pandas as pd

train_df = pd.read_csv("./HeartTrain.csv")
test_df = pd.read_csv("./HeartTest.csv")

#==== Q1 ====#
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score

X_train = train_df.drop(columns=['labels'])
y_train = train_df['labels']

# kernel=linear pr un SVM linéaire
svm = Pipeline([('scaler', StandardScaler()),('svm', SVC(kernel='linear'))])

# cross validation (10 folds et pas de shuffle)
cross_valid = KFold(n_splits=10, shuffle=False)

scores = cross_val_score(svm, X_train, y_train, cv=cross_valid)
cv_acc = scores.mean()

print(f"Cross-validation accuracy (Linear): {cv_acc:.4f}")

#==== Q2 ====#
X_train = train_df.drop(columns=['labels'])
y_train = train_df['labels']

results = {}
for d in range(2, 11):
    svm_poly = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='poly', degree=d))
    ])
    score = cross_val_score(svm_poly, X_train, y_train, cv=cross_valid).mean()
    results[d] = score
    print(f"Degré {d}: accuracy = {score:.4f}")

best_degree = max(results, key=results.get)
print(f"Le meilleur degré est : {best_degree}")  #-> Le meilleur degré est : 2 => copier Q1 et modif pipeline

#--- Soumission ---#
X_train = train_df.drop(columns=['labels'])
y_train = train_df['labels']

best_degree = 2 

svm = Pipeline([('scaler', StandardScaler()),('svm', SVC(kernel='poly', degree=best_degree))])

cross_valid = KFold(n_splits=10, shuffle=False)

scores = cross_val_score(svm, X_train, y_train, cv=cross_valid)
cv_acc = scores.mean()

print(f"Cross-validation accuracy (Poly degree {best_degree}): {cv_acc:.4f}")
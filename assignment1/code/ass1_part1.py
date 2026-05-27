import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
GLOBAL_SEED = 0

df = pd.read_csv("../data/spambase_augmented.csv", header=None)
print(df.shape) #(4601, 1186) -> 4601 rows, 1186 cols -> 

features = df.iloc[ : , :-1] #all except label
labels = df.iloc [ :, -1] #label

#Grabbed example to shuffle and split
#from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=GLOBAL_SEED)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Component 1

# decision tree with pruning
# https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning
alphas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
depths = []


print("--------------- Decision Tree with different ccp_alpha -----------------")
for alpha in alphas:
    clf_decision_tree = tree.DecisionTreeClassifier(
        random_state=GLOBAL_SEED,
        ccp_alpha=alpha,
    ) # should try with different depths to prevent overfitting maybe

    clf_decision_tree.fit(X_train, y_train)

    y_pred = clf_decision_tree.predict(X_test) #predict with trained

    acc = accuracy_score(y_test, y_pred) * 100
    train_error = (clf_decision_tree.predict(X_train) != y_train).mean() * 100
    
    
    print(f"alpha={alpha:.6f} ---- accuracy={acc:.4f}%") 
    print(f"train-error={train_error:.4f}%")

#Random forest ( no pruning )
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#

clf_forest = RandomForestClassifier(n_estimators=100, max_features="sqrt",random_state=GLOBAL_SEED) #will experiment with different n estimators, start at sqrt
clf_forest.fit(X_train, y_train)
y_pred = clf_forest.predict(X_test) #predict with trained

acc = accuracy_score(y_test, y_pred) * 100
train_error = (clf_forest.predict(X_train) != y_train).mean() * 100

print("--------------- Random Forest -----------------")
print(f"n_estimators={100} ---- accuracy={acc:.4f}%") 
print(f"train-error={train_error:.4f}%")
#Boosted decision trees https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

clf_ada = AdaBoostClassifier(n_estimators=100,learning_rate=1,random_state=GLOBAL_SEED) #will experiment with different n estimators and learning_rate
clf_ada.fit(X_train, y_train)
y_pred = clf_ada.predict(X_test) #predict with trained

acc = accuracy_score(y_test, y_pred) * 100
train_error = (clf_ada.predict(X_train) != y_train).mean() * 100

print("--------------- Adaboost -----------------")
print(f"n_estimators={100}, learning_rate={1.0} ---- accuracy={acc:.4f}%") 
print(f"train-error={train_error:.4f}%")

#defining k-fold cross-validation

def k_fold_cross_valid():
  return

#plotting
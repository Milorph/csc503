import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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
depths = [2, 4, 6, 8, 10, 12, 14]


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
    
    
    print(f"ccp_alpha={alpha:.6f} ---- accuracy={acc:.8f}%") 
    print(f"train-error={train_error:.8f}%")

print("--------------- Decision Tree with different max_depths -----------------")
for depth in depths:
    clf_decision_tree = tree.DecisionTreeClassifier(
        random_state=GLOBAL_SEED,
        ccp_alpha=1e-3,
        max_depth=depth
    ) # should try with different depths to prevent overfitting maybe

    clf_decision_tree.fit(X_train, y_train)

    y_pred = clf_decision_tree.predict(X_test) #predict with trained

    acc = accuracy_score(y_test, y_pred) * 100
    train_error = (clf_decision_tree.predict(X_train) != y_train).mean() * 100
    
    
    print(f"max_depths={depth} ---- accuracy={acc:.8f}%") 
    print(f"train-error={train_error:.8f}%")

#Random forest ( no pruning )
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#

estimators = [50,100,150,200,250,300,350,400]
max_features = ["sqrt", "log2", 50, 80, 110, 140, 170, 200]

print("--------------- Random Forest (features variation) -----------------")
for feature in max_features:
    clf_forest = RandomForestClassifier(max_features=feature,random_state=GLOBAL_SEED,  n_jobs=-1) 
    clf_forest.fit(X_train, y_train)
    y_pred = clf_forest.predict(X_test) #predict with trained

    acc = accuracy_score(y_test, y_pred) * 100
    train_error = (clf_forest.predict(X_train) != y_train).mean() * 100


    print(f"n_estimators={100}, max_features={feature} ---- accuracy={acc:.8f}%") 
    print(f"train-error={train_error:.8f}%")

print("--------------- Random Forest (n_estimators variation) -----------------")
for estimator in estimators:
    clf_forest = RandomForestClassifier(n_estimators=estimator,random_state=GLOBAL_SEED,  n_jobs=-1) 
    clf_forest.fit(X_train, y_train)
    y_pred = clf_forest.predict(X_test) #predict with trained

    acc = accuracy_score(y_test, y_pred) * 100
    train_error = (clf_forest.predict(X_train) != y_train).mean() * 100


    print(f"n_estimators={estimator} ---- accuracy={acc:.8f}%") 
    print(f"train-error={train_error:.8f}%")
#Boosted decision trees https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html

depths = [1, 2, 3, 4, 5, 6, 7]
print("--------------- adaboost (depth variation) -----------------")
for depth in depths:
    clf_ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=100,random_state=GLOBAL_SEED) 
    clf_ada.fit(X_train, y_train)
    y_pred = clf_ada.predict(X_test) #predict with trained

    acc = accuracy_score(y_test, y_pred) * 100
    train_error = (clf_ada.predict(X_train) != y_train).mean() * 100


    print(f"n_estimators={100}, max_depth={depth} ---- accuracy={acc:.8f}%") 
    print(f"train-error={train_error:.8f}%")

print("--------------- adaboost (n_estimators variation) -----------------")
for estimator in estimators:
    clf_ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),n_estimators=estimator,random_state=GLOBAL_SEED) 
    clf_ada.fit(X_train, y_train)
    y_pred = clf_ada.predict(X_test) #predict with trained

    acc = accuracy_score(y_test, y_pred) * 100
    train_error = (clf_ada.predict(X_train) != y_train).mean() * 100


    print(f"n_estimators={estimator} ---- accuracy={acc:.8f}%") 
    print(f"train-error={train_error:.8f}%")

#defining k-fold cross-validation

def k_fold_cross_valid():
  return

#plotting
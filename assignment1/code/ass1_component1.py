import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
dt_alpha_results = []
for alpha in alphas:
    clf_decision_tree = tree.DecisionTreeClassifier(
        random_state=GLOBAL_SEED,
        ccp_alpha=alpha,
    ) # should try with different depths to prevent overfitting maybe
 
    clf_decision_tree.fit(X_train, y_train)
 
    y_pred = clf_decision_tree.predict(X_test) #predict with trained
 
    train_error = (clf_decision_tree.predict(X_train) != y_train).mean() * 100
    test_error  = (clf_decision_tree.predict(X_test)  != y_test).mean() * 100
    
    
    print(f"ccp_alpha={alpha:.6f} ---- test_error={test_error:.4f}%") 
    print(f"train-error={train_error:.4f}%")
    dt_alpha_results.append({"ccp_alpha": alpha, "train_error": train_error, "test_error": test_error})
 
print("--------------- Decision Tree with different max_depths -----------------")
dt_depth_results = []
for depth in depths:
    clf_decision_tree = tree.DecisionTreeClassifier(
        random_state=GLOBAL_SEED,
        ccp_alpha=1e-3,
        max_depth=depth
    ) # should try with different depths to prevent overfitting maybe
 
    clf_decision_tree.fit(X_train, y_train)
 
    y_pred = clf_decision_tree.predict(X_test) #predict with trained
 
    test_error = (clf_decision_tree.predict(X_test)  != y_test).mean() * 100
    train_error = (clf_decision_tree.predict(X_train) != y_train).mean() * 100
    
    
    print(f"max_depths={depth} ---- test_error={test_error:.4f}%") 
    print(f"train-error={train_error:.4f}%")
    dt_depth_results.append({"max_depth": depth, "train_error": train_error, "test_error": test_error})
 
#Random forest ( no pruning )
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#
 
estimators = [1, 5, 10, 20, 50,100,150,200,250,300,350,400]
max_features = ["sqrt", "log2", 50, 80, 110, 140, 170, 200]
 
print("--------------- Random Forest (features variation) -----------------")
rf_feature_results = []
for feature in max_features:
    clf_forest = RandomForestClassifier(max_features=feature,random_state=GLOBAL_SEED,  n_jobs=-1) 
    clf_forest.fit(X_train, y_train)
    y_pred = clf_forest.predict(X_test) #predict with trained
 
    test_error = (clf_forest.predict(X_test)  != y_test).mean() * 100
    
    train_error = (clf_forest.predict(X_train) != y_train).mean() * 100
 
 
    print(f"n_estimators={100}, max_features={feature} ---- test_error={test_error:.4f}%") 
    print(f"train-error={train_error:.4f}%")
    rf_feature_results.append({"max_features": feature, "train_error": train_error, "test_error": test_error})
 
print("--------------- Random Forest (n_estimators variation) -----------------")
rf_estimator_results = []
for estimator in estimators:
    clf_forest = RandomForestClassifier(n_estimators=estimator,random_state=GLOBAL_SEED,  n_jobs=-1) 
    clf_forest.fit(X_train, y_train)
    y_pred = clf_forest.predict(X_test) #predict with trained
 
    test_error = (clf_forest.predict(X_test)  != y_test).mean() * 100
    train_error = (clf_forest.predict(X_train) != y_train).mean() * 100
 
 
    print(f"n_estimators={estimator} ---- test_error={test_error:.4f}%") 
    print(f"train-error={train_error:.4f}%")
    rf_estimator_results.append({"n_estimators": estimator, "train_error": train_error, "test_error": test_error})
#Boosted decision trees https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
 
depths = [1, 2, 3, 4, 5, 6, 7]
print("--------------- adaboost (depth variation) -----------------")
ada_depth_results = []
for depth in depths:
    clf_ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=depth), n_estimators=100,random_state=GLOBAL_SEED) 
    clf_ada.fit(X_train, y_train)
    y_pred = clf_ada.predict(X_test) #predict with trained
 
    test_error = (clf_ada.predict(X_test)  != y_test).mean() * 100
    train_error = (clf_ada.predict(X_train) != y_train).mean() * 100
 
 
    print(f"n_estimators={100}, max_depth={depth} ----  test_error={test_error:.4f}%") 
    print(f"train-error={train_error:.4f}%")
    ada_depth_results.append({"max_depth": depth, "train_error": train_error, "test_error": test_error})
 
print("--------------- adaboost (n_estimators variation) -----------------")
ada_estimator_results = []
for estimator in estimators:
    clf_ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),n_estimators=estimator,random_state=GLOBAL_SEED) 
    clf_ada.fit(X_train, y_train)
    y_pred = clf_ada.predict(X_test) #predict with trained
 
    test_error = (clf_ada.predict(X_test)  != y_test).mean() * 100
    train_error = (clf_ada.predict(X_train) != y_train).mean() * 100
 
 
    print(f"n_estimators={estimator} ---- test_error={test_error:.4f}%") 
    print(f"train-error={train_error:.4f}%")
    ada_estimator_results.append({"n_estimators": estimator, "train_error": train_error, "test_error": test_error})
    
# different training set sizes
 
 
fractions = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.0] # to be multipled with training set size
 
print("--------------- Decision Tree (training size variation) -----------------")
dt_size_results = []
for f in fractions:
    n = int(f * len(X_train))
    X_sub = X_train.iloc[:n]
    y_sub = y_train.iloc[:n]
 
    clf = tree.DecisionTreeClassifier(random_state=GLOBAL_SEED, ccp_alpha=1e-3)
    clf.fit(X_sub, y_sub)
 
    train_error = (clf.predict(X_sub) != y_sub).mean() * 100
    test_error = (clf.predict(X_test) != y_test).mean() * 100
    print(f"n={n} ---- train_error={train_error:.8f}%  test_error={test_error:.4f}%")
    dt_size_results.append({"n": n, "train_error": train_error, "test_error": test_error})
 
 
print("--------------- Random Forest (training size variation) -----------------")
rf_size_results = []
for f in fractions:
    n = int(f * len(X_train)) 
    X_sub = X_train.iloc[:n]
    y_sub = y_train.iloc[:n]
 
    clf_forest = RandomForestClassifier(n_estimators=100,random_state=GLOBAL_SEED,  n_jobs=-1)
    clf_forest.fit(X_sub, y_sub)
 
    train_error = (clf_forest.predict(X_sub) != y_sub).mean() * 100
    test_error = (clf_forest.predict(X_test) != y_test).mean() * 100
 
 
    print(f"n={n} ---- train_error={train_error:.8f}%  test_error={test_error:.4f}%")
    rf_size_results.append({"n": n, "train_error": train_error, "test_error": test_error})
 
 
print("--------------- Adaboost (training size variation) -----------------")
ada_size_results = []
for f in fractions:
    n = int(f * len(X_train))
    X_sub = X_train.iloc[:n]
    y_sub = y_train.iloc[:n]
 
    clf_ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),n_estimators=100,random_state=GLOBAL_SEED) 
    clf_ada.fit(X_sub, y_sub)
 
    train_error = (clf_ada.predict(X_sub) != y_sub).mean() * 100
    test_error = (clf_ada.predict(X_test) != y_test).mean() * 100
 
 
    print(f"n={n} ---- train_error={train_error:.8f}%  test_error={test_error:.4f}%")
    ada_size_results.append({"n": n, "train_error": train_error, "test_error": test_error})
 
#defining k-fold cross-validation
 
# pass the model, num folds, data, global seed
def k_fold_cross_valid(model_select, k, X, y, random_seed=GLOBAL_SEED):
    n = len(X)
 
    #required shuffling the indices once to be considered a k-fold-cross-valid
    rng = np.random.RandomState(random_seed)
    idx = rng.permutation(n)
 
    #size of each fold
    fold_size = n // k
    #slice each part of arr
    folds = []
    
    for i in range(k):
        start = i * fold_size #start and endpoints of each slice
        end = (i+1) * fold_size if i < k - 1 else n
        folds.append(idx[start:end])
    
    val_errors = []
    for i in range(len(folds)):
        held_out_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i]) # put together all the other folds
        model_select.fit(X.iloc[train_idx], y.iloc[train_idx])
        err = (model_select.predict(X.iloc[held_out_idx]) != y.iloc[held_out_idx]).mean()
        val_errors.append(err)
    
    return np.mean(val_errors), val_errors
 
# k-fold random forest
rf_sizes = [1, 5, 10, 25, 50, 100, 150, 200, 250, 300 ]
rf_scores = {}
for m in rf_sizes:
    print(f"[RF k-fold] n_estimators={m}")
    rf = RandomForestClassifier(n_estimators=m, random_state=GLOBAL_SEED, n_jobs=-1)
    mean_err, _ = k_fold_cross_valid(rf, 5, X_train, y_train)
    rf_scores[m] = mean_err
 
best_rf_size = min(rf_scores, key=rf_scores.get) #find best param for n_estimator
 
 
# k-fold ada
ada_sizes = [1, 5, 10, 25, 50, 100, 150, 200, 250 300]
ada_scores = {}
for m in ada_sizes:
    print(f"[AdaBoost k-fold] n_estimators={m}")
    ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),n_estimators=m, random_state=GLOBAL_SEED)
    mean_err, _ = k_fold_cross_valid(ada, 5, X_train, y_train)
    ada_scores[m] = mean_err
 
 
best_ada_size = min(ada_scores, key=ada_scores.get)

# use best params
best_rf = RandomForestClassifier(n_estimators=best_rf_size, random_state=GLOBAL_SEED, n_jobs=-1).fit(X_train, y_train)
best_ada = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), n_estimators=best_ada_size, random_state=GLOBAL_SEED).fit(X_train, y_train)
 
rf_test_err  = (best_rf.predict(X_test)  != y_test).mean() * 100
ada_test_err = (best_ada.predict(X_test) != y_test).mean() * 100

print(f"tuned RF  (n_estimators={best_rf_size}) test error: {rf_test_err:.4f}%")
print(f"tuned Ada (n_estimators={best_ada_size}) test error: {ada_test_err:.4f}%")

#plotting

#helper method using stored results
def plot_train_test(results, x_key, xlabel, title, filename, categorical=False):
    xs    = [r[x_key] for r in results]
    train = [r["train_error"] for r in results]
    test  = [r["test_error"]  for r in results]

    plt.figure()
    if categorical:                         # x is non-numeric or has awkward values
        pos = range(len(xs))
        plt.plot(pos, train, marker='o', label='train error')
        plt.plot(pos, test,  marker='s', label='test error')
        plt.xticks(pos, [str(x) for x in xs], rotation=45)
    else:                                   # x is clean numeric
        plt.plot(xs, train, marker='o', label='train error')
        plt.plot(xs, test,  marker='s', label='test error')
    plt.xlabel(xlabel)
    plt.ylabel("error (%)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

#two hyperparameters per method plots
plot_train_test(dt_alpha_results,  "ccp_alpha", "ccp_alpha", "Decision tree: error vs ccp_alpha", "dt_alpha.png", categorical=True)
plot_train_test(dt_depth_results,  "max_depth", "max depth", "Decision tree: error vs max depth", "dt_depth.png")

plot_train_test(rf_feature_results, "max_features", "max_features", "Random forest: error vs max_features", "rf_features.png", categorical=True)
plot_train_test(rf_estimator_results, "n_estimators", "number of trees", "Random forest: error vs number of trees", "rf_estimators.png")

plot_train_test(ada_depth_results, "max_depth", "weak-learner max depth", "AdaBoost: error vs weak-learner depth", "ada_depth.png")
plot_train_test(ada_estimator_results, "n_estimators", "number of iterations", "AdaBoost: error vs number of iterations", "ada_estimators.png")

# training-set-size curves plots
plot_train_test(dt_size_results,  "n", "training set size", "Decision tree: error vs training size", "dt_size.png")
plot_train_test(rf_size_results,  "n", "training set size", "Random forest: error vs training size", "rf_size.png")
plot_train_test(ada_size_results, "n", "training set size", "AdaBoost: error vs training size", "ada_size.png")

#CV error vs ensemble size plots
plt.figure()
plt.plot(list(rf_scores.keys()),  [v*100 for v in rf_scores.values()],  marker='o', label='random forest')
plt.plot(list(ada_scores.keys()), [v*100 for v in ada_scores.values()], marker='s', label='AdaBoost stumps')
plt.xlabel("ensemble size"); plt.ylabel("5-fold CV error (%)")
plt.title("Part II: cross-validation error vs ensemble size")
plt.legend(); plt.grid(True, alpha=0.3)
plt.savefig("cv_vs_size.png", dpi=150, bbox_inches="tight")
plt.close()
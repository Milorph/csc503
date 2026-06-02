## Data Mining - Summer 2026 (author: Nishant Mehta)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split 
from sklearn.utils import check_random_state  
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GLOBAL_SEED = 0

## THE ACTUAL STARTER CODE YOU SHOULD GRAB BEGINS BELOW

## Assumptions
#    - n_samples is the number of examples
#    - n_samples_bootstrap is the number of samples in each bootstrap sample
#      (this should be equal to n_samples)
#    - rf is a random forest, obtained via a call to
#      RandomForestClassifier(...) in scikit-learn


def oob_error(rf, X, y):
    #given starter, replacing dummy data
    X_arr = np.asarray(X)
    y = np.asarray(y)
    n_samples = len(X_arr)
    n_samples_bootstrap = n_samples
    unsampled_indices_for_all_trees= []
    classes = rf.classes_
    voting = np.zeros((n_samples, len(classes)))
    
    for estimator in rf.estimators_:
        random_instance = check_random_state(estimator.random_state)
        sample_indices = random_instance.randint(0, n_samples, n_samples_bootstrap)
        sample_counts = np.bincount(sample_indices, minlength = n_samples)
        unsampled_mask = (sample_counts == 0)
        indices_range = np.arange(n_samples)
        unsampled_indices = indices_range[unsampled_mask]
        unsampled_indices_for_all_trees += [unsampled_indices]
        
    for tree, idx in zip(rf.estimators_ , unsampled_indices_for_all_trees):
        #skip if not oob
        if len(idx) == 0:
            continue
        #predicting for that unsampled
        preds = tree.predict(X_arr[idx])
        #add 1 to the class label for the idx, astype fixes floating points issues
        np.add.at(voting, (idx, preds.astype(int)), 1)
    oob_pred_cols = np.argmax(voting, axis = 1) # column max of the class
    oob_preds = classes[oob_pred_cols] #mapping 1d predicitons to actual class labels
    
    covered = voting.sum(axis=1) > 0 #skip over ones with no votes + make a True/False mask

    err = np.mean(oob_preds[covered] != y[covered])
    
    return err
    

df = pd.read_csv("../data/spambase_augmented.csv", header=None)
print(df.shape) #(4601, 1186) -> 4601 rows, 1186 cols -> 
 
features = df.iloc[ : , :-1] #all except label
labels = df.iloc [ :, -1] #label
 
#Grabbed example to shuffle and split
#from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=GLOBAL_SEED)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
clf_forest = RandomForestClassifier(oob_score=True, random_state=GLOBAL_SEED,  n_jobs=-1) 
clf_forest.fit(X_train, y_train)
scikit_oob_error = 1 - clf_forest.oob_score_

print(f"scikit-learn oob_error: {scikit_oob_error}")
print(f"implemented oob_error: {oob_error(clf_forest, X_train, y_train)}")

#plotting

rf = RandomForestClassifier(warm_start=True, random_state=GLOBAL_SEED, n_jobs=-1)
tree_counts = list(range(1, 30)) + [40, 50, 75, 100, 150]
oob_curve = []
for m in tree_counts:
    rf.n_estimators = m
    rf.fit(X_train, y_train)   
    oob_curve.append(oob_error(rf, X_train, y_train))
    
plt.plot(tree_counts, oob_curve, color='green', linestyle='--', marker='o')
plt.title("OOB error vs number of trees")
plt.xlabel("num trees")
plt.ylabel("OOB error")
plt.savefig("oob_vs_trees.png", dpi=150, bbox_inches="tight")
plt.show()


## Result:
#    unsampled_indices_for_all_trees is a list with one element for each tree
#    in the forest. In more detail, the j'th element is an array of the example
#    indices that were \emph{not} used in the training of j'th tree in the
#    forest. For examle, if the 1st tree in the forest was trained on a
#    bootstrap sample that was missing only the first and seventh training
#    examples (corresponding to indices 0 and 6), and if the last tree in the
#    forest was trained on a boostrap sample that was missing the second,
#    third, and sixth training examples (indices 1, 2, and 5), then
#    unsampled_indices_for_all_trees would begin like:  
#        [array([0, 6]),
#         ...
#         array([1, 2, 5])]


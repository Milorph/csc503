import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn import tree
from sklearn.metrics import accuracy_score
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
alphas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
depths = []

for alpha in alphas:
    clf = tree.DecisionTreeClassifier(
        random_state=GLOBAL_SEED,
        ccp_alpha=alpha
    ) # should try with different depths to prevent overfitting maybe

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test) #train model

    acc = accuracy_score(y_test, y_pred) * 100
    train_error = (clf.predict(X_train) != y_train).mean() * 100
    

    print(f"alpha={alpha:.6f} ---- accuracy={acc:.4f}%") 
    print(f"train-error={train_error:.4f}%")

#Random forest ( no pruning )

#Boosted decision trees

#defining k-fold cross-validation

def k_fold_cross_valid():
  return

#plotting
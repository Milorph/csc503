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
alphas = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 0.003, 0.002]

for alpha in alphas:
    clf = tree.DecisionTreeClassifier(
        random_state=GLOBAL_SEED,
        ccp_alpha=alpha
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(f"alpha={alpha:.6f} -> accuracy={acc:.4f}") # accuracy was 

#Random forest ( no pruning )

#Boosted decision trees

#defining k-fold cross-validation

def k_fold_cross_valid():
  return
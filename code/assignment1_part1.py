import pandas as pd
from sklearn.model_selection import train_test_split

GLOBAL_SEED = 0

df = pd.read_csv("../data/spambase_augmented.csv", header=None)
print(df.shape) #(4601, 1186) -> 4601 rows, 1186 cols -> 

features = df.iloc[ : , :-1] #all except label
labels = df.iloc [ :, -1] #label

#Grabbed example to shuffle and split
#from https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=GLOBAL_SEED)
print(X_train.shape(), X_test.shape(), y_train.shape(), y_test.shape())
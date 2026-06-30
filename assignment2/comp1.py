import os, gzip
import numpy as np


#GLOBAL VARIABLES

DATA_DIR = "assignment2/data/fashion-mnist-master/data/fashion"
SEED     = 0
P_FLIP = 0.2 #between 0.1 and 0.3 for p
rng = np.random.default_rng(SEED)


def load_mnist(path, kind="train"):
    #from the repo's utils/mnist_reader.py)
    labels_path = os.path.join(path, f"{kind}-labels-idx1-ubyte.gz")
    images_path = os.path.join(path, f"{kind}-images-idx3-ubyte.gz")
    with gzip.open(labels_path, "rb") as f:
        labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    with gzip.open(images_path, "rb") as f:
        images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels
  
def load_data():
    #load full MNIST
    Xtr_full, ytr_full = load_mnist(DATA_DIR, kind="train")
    Xte_full, yte_full = load_mnist(DATA_DIR, kind="t10k")
 
    #keep only classes 5 and 7 while remapping 5 -> 0, 7 -> 1
    def keep_57(X, y):
        mask = (y == 5) | (y == 7)
        return X[mask], np.where(y[mask] == 5, 0, 1)
    Xtr, ytr = keep_57(Xtr_full, ytr_full)
    Xte, yte = keep_57(Xte_full, yte_full)
 
    #normalize inputs to [0, 1]
    Xtr = Xtr.astype(np.float64) / 255.0
    Xte = Xte.astype(np.float64) / 255.0
 
    # add noise
    flip = rng.random(len(ytr)) < P_FLIP
    ytr_noisy = ytr.copy()
    ytr_noisy[flip] = 1 - ytr_noisy[flip]
 
    return Xtr, ytr_noisy, Xte, yte, flip.sum()

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
  
  
  
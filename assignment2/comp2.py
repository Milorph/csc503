import os, gzip
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate
from sklearn.svm import SVC

DATA_DIR = "data/fashion-mnist-master/data/fashion"
SEED = 0
rng = np.random.default_rng(SEED)
 
#load data with no noise, classes 5->0, 7->1, normalize to [0,1] same as comp1
def load_mnist(path, kind="train"):
    lp = os.path.join(path, f"{kind}-labels-idx1-ubyte.gz")
    ip = os.path.join(path, f"{kind}-images-idx3-ubyte.gz")
    with gzip.open(lp, "rb") as f: labels = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    with gzip.open(ip, "rb") as f: images = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
    return images, labels
  
def keep_57(X, y):
    m = (y == 5) | (y == 7)
    return X[m], np.where(y[m] == 5, 0, 1)
 
Xtr_full, ytr_full = load_mnist(DATA_DIR, "train")
Xte_full, yte_full = load_mnist(DATA_DIR, "t10k")
Xtr_full, ytr_full = keep_57(Xtr_full, ytr_full)
Xte, yte = keep_57(Xte_full, yte_full)
Xtr_full = Xtr_full.astype(np.float64) / 255.0
Xte = Xte.astype(np.float64) / 255.0

# Nevermind im not gonna do it
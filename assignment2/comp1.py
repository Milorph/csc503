import os, gzip
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier



#GLOBAL VARIABLES

DATA_DIR = "data/fashion-mnist-master/data/fashion"
SEED = 0
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

#print checks (forgot to add to commit)
Xtr, ytr, Xte, yte, n_flipped = load_data()
print("train:", Xtr.shape, " test:", Xte.shape)
print("train class counts (after noise):", np.bincount(ytr))
print("test  class counts:", np.bincount(yte))
print(f"flipped {n_flipped} / {len(ytr)} training labels (p={P_FLIP})")
print("pixel range:", Xtr.min(), "to", Xtr.max())


#defining k-fold cross-validation
 
# pass the model, num folds, data, global seed
def k_fold_cross_valid(model_select, k, X, y, random_seed=SEED):
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
        model_select.fit(X[train_idx], y[train_idx]) #had to change iloc because it doesn't fit this dataset
        err = (model_select.predict(X[held_out_idx]) != y[held_out_idx]).mean()
        val_errors.append(err)
    
    return np.mean(val_errors), val_errors
  
  
#make the linear svm

K = 5
N_TRAIN = len(Xtr)   # 12000 used for the alpha = 1/(N*C) conversion 

def make_linear_svm(C):
    alpha = 1.0 / (N_TRAIN * C)
    return SGDClassifier(loss="hinge", alpha=alpha, max_iter=2000, tol=1e-4, random_state=SEED)

#tune C with my k-fold CV, using logspace because its much better linear grid
C_grid_cv = np.logspace(-5, 4, 10)
cv_means = []
for C in C_grid_cv:
    mean_err, _ = k_fold_cross_valid(make_linear_svm(C), K, Xtr, ytr)
    cv_means.append(mean_err)
    print(f"  C={C:>10.4g}   {K}-fold CV error = {mean_err:.4f}")
best_C = float(C_grid_cv[int(np.argmin(cv_means))])
print(f">>> recorded optimal linear-SVM C = {best_C:.4g}")

#train/test error over a wider grid, using train and test sets
C_grid_plot = np.logspace(-6, 4, 21)
tr_err, te_err = [], []
for C in C_grid_plot:
  m = make_linear_svm(C); m.fit(Xtr, ytr)
  tr_err.append((m.predict(Xtr) != ytr).mean())
  te_err.append((m.predict(Xte) != yte).mean())

#Plottings
plt.figure()
plt.semilogx(C_grid_plot, tr_err, "o-", label="training error (noisy labels)")
plt.semilogx(C_grid_plot, te_err, "s-", label="test error")
plt.axvline(best_C, ls="--", color="gray", label=f"CV-chosen C={best_C:.2g}")
plt.xlabel("regularization parameter C")
plt.ylabel("classification error")
plt.title("Linear SVM: train/test error vs C")
plt.legend(); plt.tight_layout()
plt.savefig("results/linear_svm.png", dpi=150)
plt.show()
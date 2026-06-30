import os, gzip
import numpy as np
import matplotlib.pyplot as plt

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier



#GLOBAL VARIABLES

DATA_DIR = "data/fashion-mnist-master/data/fashion" #REMINDER FOR MYSELF: add the instructions to make a folder data in same directory and extract the zip contents into it
SEED = 0
P_FLIP = 0.2 #between 0.1 and 0.3 for p
rng = np.random.default_rng(SEED)

#from the repo's utils/mnist_reader.py)
def load_mnist(path, kind="train"):
    
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
    Xte_full, yte_full = load_mnist(DATA_DIR, kind="t10k") #loads 10000
 
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

#print checks
Xtr, ytr, Xte, yte, n_flipped = load_data()
print("train:", Xtr.shape, " test:", Xte.shape)
print("train class counts (after noise):", np.bincount(ytr))
print("test  class counts:", np.bincount(yte))
print(f"flipped {n_flipped} / {len(ytr)} training labels (p={P_FLIP})")
print("pixel range:", Xtr.min(), "to", Xtr.max())


PER_CLASS = 2000 #The entire 12000 was super slow so this is configurable for now, 3000 still too slow, 1000 was allowed so I'm doing it
if PER_CLASS is not None:
    idx = np.concatenate([rng.permutation(np.where(ytr == c)[0])[:PER_CLASS] for c in (0, 1)])
    idx = rng.permutation(idx)
    Xtr, ytr = Xtr[idx], ytr[idx]
print("training set reduced to:", Xtr.shape)

#defining k-fold cross-validation FROM ass1
 
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
    print(f"  C={C:.4f}   {K}-fold CV error = {mean_err:.4f}")
best_C = float(C_grid_cv[int(np.argmin(cv_means))])
print(f"----- recorded optimal linear-SVM C = {best_C:.4f}")

#train/test error over a wider grid, using train and test sets
C_grid_plot = np.logspace(-6, 4, 21)
tr_err_lin, te_err_lin = [], [] #renamed
for C in C_grid_plot:
  m = make_linear_svm(C); m.fit(Xtr, ytr)
  tr_err_lin.append((m.predict(Xtr) != ytr).mean())
  te_err_lin.append((m.predict(Xte) != yte).mean())
  

#gaussian kernel
#reference: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

gamma_grid = np.logspace(-4, 0, 6)  
C_grid     = np.logspace(-1, 3, 5) 

#for each gamma, tune C by k-fold CV
tuned, tuned_cv = [], []
for g in gamma_grid:
    errs = []
    for C in C_grid:
        e, _ = k_fold_cross_valid(SVC(kernel="rbf", gamma=g, C=C), K, Xtr, ytr)
        errs.append(e)
    j = int(np.argmin(errs))
    tuned.append((g, float(C_grid[j])))
    tuned_cv.append(errs[j]) # one tuned (gamma, C_gamma) SVM per gamma
    print(f"gamma={g:.4f} , tuned C={C_grid[j]:.4f} , (CV err {errs[j]:.4f})")

#pick the best gamma by comparing the tu CV scores
bi = int(np.argmin(tuned_cv))
best_gamma, best_Cg = tuned[bi]
print(f"recorded optimal Gaussian SVM: gamma={best_gamma:.4g}, C={best_Cg:.4g}")

#for each tuned SVM, train on full train, record train and test error (renamed it becuz of similar variable names)
tr_err_g, te_err_g = [], []
for (g, C) in tuned:
    m = SVC(kernel="rbf", gamma=g, C=C).fit(Xtr, ytr)
    tr_err_g.append((m.predict(Xtr) != ytr).mean())
    te_err_g.append((m.predict(Xte) != yte).mean())



#Neural net next

#https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

def make_mlp(hidden, activation, max_iter=200, alpha=1e-4):
    return MLPClassifier(hidden_layer_sizes=hidden, activation=activation, alpha=alpha, max_iter=max_iter, solver="adam", random_state=SEED)

#tune structure + nonlinearity with my k-fold CV
nn_configs = [((50,), "relu"), ((100,), "relu"), ((50,), "tanh"), ((100,), "tanh"), ((50, 50), "relu")]

nn_cv = []
for (hidden, act) in nn_configs:
    e, _ = k_fold_cross_valid(make_mlp(hidden, act), K, Xtr, ytr)
    nn_cv.append(e)
    print(f"hidden={hidden}, act={act} CV err = {e:.4f}")
bi_nn = int(np.argmin(nn_cv))
best_hidden, best_act = nn_configs[bi_nn]
print(f"recorded optimal NN: hidden={best_hidden}, activation={best_act}")

#Testing the neural net on the experiments

#vary number of hidden nodes
node_grid = [5, 10, 25, 50, 100, 200]
tr_err_nnA, te_err_nnA = [], []
for h in node_grid:
    m = make_mlp((h,), best_act).fit(Xtr, ytr)
    tr_err_nnA.append((m.predict(Xtr) != ytr).mean())
    te_err_nnA.append((m.predict(Xte) != yte).mean())

#vary number of training epochs
epoch_grid = [1, 2, 5, 10, 25, 50, 100, 200]
tr_err_nnB, te_err_nnB = [], []
for mi in epoch_grid:
    m = make_mlp(best_hidden, best_act, max_iter=mi).fit(Xtr, ytr)
    tr_err_nnB.append((m.predict(Xtr) != ytr).mean())
    te_err_nnB.append((m.predict(Xte) != yte).mean())
    
#compare the svm, gaussian kernel and neural net (using the fine tuned, just call the helpers with the best param)





#Plottings

os.makedirs("results", exist_ok=True)

plt.figure()
plt.semilogx(C_grid_plot, tr_err_lin, "o-", label="training error (noisy labels)")
plt.semilogx(C_grid_plot, te_err_lin, "s-", label="test error")
plt.axvline(best_C, ls="--", color="gray", label=f"CV-chosen C={best_C:.2g}")
plt.xlabel("regularization parameter C")
plt.ylabel("classification error")
plt.title("Linear SVM: train/test error vs C")
plt.legend(); plt.tight_layout()
plt.savefig("results/linear_svm.png", dpi=150)
plt.show()

plt.figure()
plt.semilogx(gamma_grid, tr_err_g, "o-", label="training error (noisy labels)")
plt.semilogx(gamma_grid, te_err_g, "s-", label="test error")
plt.axvline(best_gamma, ls="--", color="gray", label=f"CV-chosen gamma={best_gamma:.2g}")
plt.xlabel("Gaussian-kernel scale gamma")
plt.ylabel("classification error")
plt.title("Gaussian-kernel SVM (C tuned per gamma): error vs gamma")
plt.legend(); plt.tight_layout()
plt.savefig("results/gaussian_svm.png", dpi=150)
plt.show()

plt.figure()
plt.plot(node_grid, tr_err_nnA, "o-", label="training error (noisy labels)")
plt.plot(node_grid, te_err_nnA, "s-", label="test error")
plt.xlabel("number of nodes in hidden layer")
plt.ylabel("classification error")
plt.title(f"NN: vary hidden-layer width (activation={best_act})")
plt.legend(); plt.tight_layout()
plt.savefig("results/nn_width.png", dpi=150)
plt.show()

plt.figure()
plt.plot(epoch_grid, tr_err_nnB, "o-", label="training error (noisy labels)")
plt.plot(epoch_grid, te_err_nnB, "s-", label="test error")
plt.xlabel("max training epochs (max_iter)")
plt.ylabel("classification error")
plt.title(f"NN: vary training epochs (hidden={best_hidden}, act={best_act})")
plt.legend(); plt.tight_layout()
plt.savefig("results/nn_epochs.png", dpi=150)
plt.show()
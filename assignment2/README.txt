Install requirements with:
    pip install numpy matplotlib scikit-learn
 
 
IMPORTANT FOR GRADER:

comp1.py expects the Fashion-MNIST data at:
 
    data/fashion-mnist-master/data/fashion/
 
Steps to reproduce:
  1. In the same directory as comp1.py, create a folder called "data".
  2. Download the dataset:
        https://github.com/zalandoresearch/fashion-mnist/archive/master.zip
  3. Extract the zip into the "data" folder, so that the four .gz files end up at
     the path shown above.
 

Run code using:
    python comp1.py

Experiments done:
  The script prints the cross-validation scores, the selected hyperparameters,
  and the final test errors (with 95% confidence intervals) to the console.
  It also writes the five figures used in the report to a "results/" folder,
  which it creates automatically:
      results/linear_svm.png
      results/gaussian_svm.png
      results/nn_width.png
      results/nn_epochs.png
      results/comparison.png
 
Note: To keep runtime reasonable (and allowed in the assignment) the training set is reduced to 2000 examples
per class (4000 total), controlled by the PER_CLASS variable near the top of
the script. The linear-SVM train/test-vs-C plot uses the full 12000-example
training set. 
 
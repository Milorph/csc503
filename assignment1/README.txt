==============================================================
 Data Mining - Assignment 1 (CSC 503)
 Decision Trees, Random Forests, AdaBoost, and an OOB Estimate
 Author: Robert
==============================================================

------------------------------------------------------------
 1. HOW TO RUN
------------------------------------------------------------
  Run BOTH scripts from inside the 'code' directory so the
  "../data/..." path resolves correctly:

  cd code
  python ass1_component1.py
  python oob_error.py


------------------------------------------------------------
 2. ATTRIBUTION
------------------------------------------------------------
  - The learning algorithms (DecisionTreeClassifier,
    RandomForestClassifier, AdaBoostClassifier) are from the
    scikit-learn library. Relevant scikit-learn documentation URLs
    are cited inline as comments in ass1_component1.py
    (decision trees, minimal cost-complexity pruning, random
    forests, AdaBoost) and the train/test split example is adapted
    from the scikit-learn train_test_split documentation.

  - oob_error.py is built on the OOB starter code provided by CSC 503 (Prof. Nishant Mehta)

  - spambase_augmented.csv is in the zip file, in the data folder

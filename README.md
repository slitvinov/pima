# Pima Indians Diabetes Database

J. W. Smith, J. E. Everhart, W. C. Dickson, W. C. Knowler and
R. S. Johannes, "Using the ADAP learning algorithm to forecast the
onset of diabetes mellitus",
Proc. Annu. Symp. Comput. Appl. Med. Care, pp. 261-265, Nov. 1988.

- https://www.kaggle.com/uciml/pima-indians-diabetes-database
- https://www.openml.org/d/37
- https://datahub.io/machine-learning/diabetes
- https://www.cs.waikato.ac.nz/ml/weka/arff.html

# Install

<pre>
python3 -m pip install arff
</pre>

# Data

<pre>
1. Number of times pregnant
2. Plasma Glucose Concentration at 2 Hours in an Oral
Glucose Tolerance Test (GTIT)
3. Diastolic Blood Pressure (mm Hg)
4. Triceps Skin Fold Thickness (mm)
5. 2-Hour Serum Insulin Uh/ml)
6. Body Mss Index (Weight in kg / (Height in in))
7. Diabetes Pedigree Function
8. Age (years)
</pre>

# Models

<pre>
classifiers = {
    "Logistic": LogisticRegression('none'),
    "Linear SVM": SVC(kernel="linear", probability=True, random_state=0),
    "RBF SVM" : SVC(gamma=2, C=1),
    "GPC": GaussianProcessClassifier(),
    "Decision Tree" : DecisionTreeClassifier(),
    "Random Forest" : RandomForestClassifier(max_depth=10, n_estimators=100, max_features=2),
    "KNeighbors" : KNeighborsClassifier(),
    "Neural Net" : MLPClassifier(alpha=1, max_iter=1000),
    "AdaBoost" : AdaBoostClassifier(),
    "Naive Bayes" : GaussianNB(),
    "QDA" : QuadraticDiscriminantAnalysis()
}
</pre>
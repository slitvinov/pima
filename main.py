import arff
import matplotlib.pyplot as plt
import numpy as np
import sklearn.discriminant_analysis
import sklearn.ensemble
import sklearn.gaussian_process
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.naive_bayes
import sklearn.neural_network
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm

models = \
    (sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(), "QDA"), \
    (sklearn.ensemble.AdaBoostClassifier(), "AdaBoost"), \
    (sklearn.ensemble.RandomForestClassifier(max_depth=10, n_estimators=100), \
     "RForest"), \
    (sklearn.gaussian_process.GaussianProcessClassifier(), "GPC"), \
    (sklearn.linear_model.LogisticRegression(), "Logistic"), \
    (sklearn.naive_bayes.GaussianNB(), "NBayes"), \
    (sklearn.neighbors.KNeighborsClassifier(), "KNeighbors"), \
    (sklearn.neural_network.MLPClassifier(alpha=1,
                                          max_iter=1000), "NN"), \
    (sklearn.svm.SVC(gamma=2, C=1), "RBF SVM"), \
    (sklearn.svm.SVC(kernel="linear", probability=True,
                     random_state=123456), "Linear SVM"), \
    (sklearn.tree.DecisionTreeClassifier(), "DTree"), \

X = []
y = []
for row in arff.load('pima.arff'):
    x = []
    n = len(row)
    for i in range(n - 1):
        x.append(row[i])
    X.append(x)
    y.append(0 if row[n - 1] == 'tested_negative' else 1)

X0, X1, y0, y1 = sklearn.model_selection.train_test_split(X,
                                                          y,
                                                          train_size=576,
                                                          random_state=123456,
                                                          stratify=y)

scaler = sklearn.preprocessing.StandardScaler()
error0 = []
error1 = []
for model, name_model in models:
    pipeline = sklearn.pipeline.Pipeline([('scaler', scaler),
                                          ('model', model)])
    pipeline.fit(X0, y0)
    for X, y, error, name_set in (X0, y0, error0, 'train'), (X1, y1, error1,
                                                             'test'):
        yp = pipeline.predict(X)
        C = sklearn.metrics.confusion_matrix(y, yp)
        fp = C[1, 0] / len(y)
        fn = C[0, 1] / len(y)
        error.append((fp, fn))

fig, ax = plt.subplots()
ax.plot(*np.transpose(error1), 'o')
plt.xlabel("false positive rate")
plt.ylabel("false negative rate")
for (fp, fn), (model, name_model) in zip(error1, models):
    ax.annotate(name_model, (fp, fn))
plt.savefig('rate.png')

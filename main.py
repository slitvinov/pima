import arff
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

for model, name_model in \
    (sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(), "QDA"), \
    (sklearn.ensemble.AdaBoostClassifier(), "AdaBoost"), \
    (sklearn.ensemble.RandomForestClassifier(max_depth=10, n_estimators=100), "Random Forest"), \
    (sklearn.gaussian_process.GaussianProcessClassifier(), "GPC"), \
    (sklearn.linear_model.LogisticRegression(), "Logistic"),  \
    (sklearn.naive_bayes.GaussianNB(), "Naive Bayes"), \
    (sklearn.neighbors.KNeighborsClassifier(), "KNeighbors"), \
    (sklearn.neural_network.MLPClassifier(alpha=1, max_iter=1000), "Neural Net"), \
    (sklearn.svm.SVC(gamma=2, C=1), "RBF SVM"), \
    (sklearn.svm.SVC(kernel="linear", probability=True, random_state=123456), "Linear SVM"), \
    (sklearn.tree.DecisionTreeClassifier(), "Decision Tree"), \
    :
    print(name_model)
    pipeline = sklearn.pipeline.Pipeline([('scaler', scaler),
                                          ('model', model)])
    pipeline.fit(X0, y0)
    for X, y, name_set in (X0, y0, 'train'), (X1, y1, 'test'):
        yp = pipeline.predict(X)
        C = sklearn.metrics.confusion_matrix(y, yp)
        print(C)
    print("")

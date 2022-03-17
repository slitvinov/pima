import arff
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

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

scaler = sklearn.preprocessing.StandardScaler().fit(X0)
model = sklearn.linear_model.LogisticRegression()
pipeline = sklearn.pipeline.Pipeline([('scaler', scaler), ('model', model)])

pipeline.fit(X0, y0)

for X, y, name in (X0, y0, 'train'), (X1, y1, 'test'):
    print(name)
    yp = pipeline.predict(X)
    C = sklearn.metrics.confusion_matrix(y, yp)
    print(C)

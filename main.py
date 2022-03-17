import arff
import sklearn.model_selection

X = [ ]
y = [ ]
for row in arff.load('pima.arff'):
    x = [ ]
    n = len(row)
    for i in range(n - 1):
        x.append(row[i])
    X.append(x)
    y.append(0 if row[n - 1] == 'tested_negative' else 1)

X0, X1, y0, y1 = sklearn.model_selection.train_test_split(X, y, test_size=576)



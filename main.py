import arff

X = [ ]
y = [ ]
for row in arff.load('pima.arff'):
    x = [ ]
    n = len(row)
    for i in range(n - 1):
        x.append(row[i])
    X.append(x)
    y.append(0 if row[n - 1] == 'tested_negative' else 1)

print(y)

import arff

for row in arff.load('pima.arff'):
    x = row
    print(x[0])

from sklearn import svm
from sklearn import datasets
import pandas
import numpy as np
import matplotlib.pyplot as plt



clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)  





dataset = pandas.DataFrame(iris.data, columns=iris.feature_names)
print(dataset.head(20))
print("==========")
print(dataset.describe())


plt.figure()

#print(dataset['Species'].unique())
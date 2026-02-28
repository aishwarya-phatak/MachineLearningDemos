#import sklearn as skl
#import sklearn.datasets as datasets

from sklearn import datasets
from sklearn.datasets import load_iris     #to get iris dataset from all datasets

#installtion of sklearn
#getting datasets

print(dir(datasets))
iris_ds = datasets.load_iris()
data_iris_ds_keys = iris_ds.keys()
print(data_iris_ds_keys)

data_iris = iris_ds.data
print(data_iris)

data_iris_filename = iris_ds.filename
print(data_iris_filename)
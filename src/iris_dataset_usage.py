#import sklearn as skl
#import sklearn.datasets as datasets
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris     #to get iris dataset from all datasets
#dataset used for classification
#installtion of sklearn
#getting datasets

print(dir(datasets))
iris_ds = datasets.load_iris()
data_iris_ds_keys = iris_ds.keys()
print(data_iris_ds_keys)

data_iris = iris_ds.data
print(data_iris)

#data_iris_filename = iris_ds.filename
#print(data_iris_filename)
data_iris_features = iris_ds.feature_names
for key in data_iris_features:
    print(key)

data_iris_targetNames = iris_ds.target_names
for name in data_iris_targetNames:
    print(name)

df = pd.DataFrame(data_iris)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df)

labels = kmeans.labels_

plt.scatter(data_iris[:,2],data_iris[:,3],c=labels)
plt.title("iris dataset clustering")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()
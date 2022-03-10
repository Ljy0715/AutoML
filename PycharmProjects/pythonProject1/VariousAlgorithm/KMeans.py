import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import classification_report

hr_data = pd.read_csv('data/hr.csv', header=0)
hr_data.head()
print(" Data Set Shape ", hr_data.shape)
print(list(hr_data.columns))
print(" Sample Data", hr_data.head())

data_trnsf = pd.get_dummies(hr_data, columns=['salary', 'sales'])
data_trnsf.columns
X = data_trnsf.drop('left', axis=1)
X.columns

X_train, X_test, Y_train, Y_test = train_test_split(X, data_trnsf.left, test_size=0.3, random_state=42)
print(X_train)

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(data_trnsf)
cluster_labels = kmeans.fit_predict(data_trnsf)

centroid = kmeans.cluster_centers_
labels = kmeans.labels_
print(centroid)
print(labels)
silhouette_avg = silhouette_score(data_trnsf, cluster_labels)
print("For n_clusters =", n_clusters, "The average silhouette_score is : ", silhouette_avg)

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pprint import pprint
from sklearn.datasets._samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from unsupervisedAutoMl import Unsupervised_AutoMl

sns.set_context('poster')

sns.set_color_codes()

plot_kwargs_src = {'s': 10, 'linewidths': 0.1}

centers = [[2, 1], [-1.5, -1], [1, -1], [-2, 2]]
cluster_std = [0.4, 0.5, 0.6, 0.5]

X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=cluster_std, random_state=53)

plt.scatter(X[:, 0], X[:, 1], **plot_kwargs_src)
plt.show()

estimators = [{'estimator': KMeans, 'args': (), 'kwargs': {'n_clusters': 4}}]
unsupervised_learner = Unsupervised_AutoMl(estimators)

print(unsupervised_learner.estimators)
predictions, performance_metrics = unsupervised_learner.fit_predict(X, y)
pprint(performance_metrics)

plot_kwargs = {'s': 12, 'linewidths': 0.1}
unsupervised_learner.plot_clusters(KMeans, X, unsupervised_learner.predictions[0]['KMeans'], plot_kwargs)


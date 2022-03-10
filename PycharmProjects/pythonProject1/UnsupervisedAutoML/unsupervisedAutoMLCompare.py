import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, estimate_bandwidth, SpectralClustering
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pprint import pprint
from sklearn.datasets._samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
from unsupervisedAutoMl import Unsupervised_AutoMl
from hdbscan import HDBSCAN

sns.set_context('poster')

sns.set_color_codes()

plot_kwargs_src = {'s': 10, 'linewidths': 0.1}

centers = [[2, 1], [-1.5, -1], [1, -1], [-2, 2]]
cluster_std = [0.4, 0.5, 0.6, 0.5]

# X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=cluster_std, random_state=53)
X, y = make_blobs(n_samples=2000, centers=5, cluster_std=[1.7, 0.6, 0.8, 1.0, 1.2], random_state=220)

plt.scatter(X[:, 0], X[:, 1], **plot_kwargs_src)
# plt.show()

# estimators = [{'estimator': KMeans, 'args': (), 'kwargs': {'n_clusters': 4}}]
# unsupervised_learner = Unsupervised_AutoMl(estimators)
#
# print(unsupervised_learner.estimators)
# predictions, performance_metrics = unsupervised_learner.fit_predict(X, y)


bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=100)
estimators = [{'estimator': KMeans, 'args': (), 'kwargs': {'n_clusters': 5}},
              {'estimator': DBSCAN, 'args': (), 'kwargs': {'eps': 0.5}},
              {'estimator': AgglomerativeClustering, 'args': (), 'kwargs': {'n_clusters': 5, 'linkage': 'ward'}},
              {'estimator': MeanShift, 'args': (), 'kwargs': {'cluster_all': False, 'bandwidth': bandwidth,
                                                              'bin_seeding': True}},
              {'estimator': SpectralClustering, 'args': (), 'kwargs': {'n_clusters': 5}},
              {'estimator': HDBSCAN, 'args': (), 'kwargs': {'min_cluster_size': 15}}]
unsupervised_learner = Unsupervised_AutoMl(estimators)
predictions, performance_metrics = unsupervised_learner.fit_predict(X, y)

pprint(predictions)

pprint(performance_metrics)

plot_kwargs = {'s': 12, 'linewidths': 0.1}
unsupervised_learner.plot_all_clusters(estimators, unsupervised_learner.predictions, X, plot_kwargs)

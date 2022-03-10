from unsupervisedAutoMl import Unsupervised_AutoMl
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pprint import pprint
from sklearn.datasets._samples_generator import make_blobs

plot_kwargs = {'s': 10, 'linewidths': 0.1}
X, y = make_blobs(n_samples=2000, centers=5, cluster_std=[1.7, 0.6, 0.8, 1.0, 1.2], random_state=220)
plt.scatter(X[:, 0], X[:, 1], **plot_kwargs)
plt.show()

estimators = [{'estimator': KMeans, 'args': (), 'kwargs': {'n_clusters': 4}}]
unsupervised_learner = Unsupervised_AutoMl(estimators)
predictions, performance_metrics = unsupervised_learner.fit_predict(X, y)
pprint(performance_metrics)

plot_kwargs = {'s': 12, 'linewidths': 0.1}
unsupervised_learner.plot_clusters(KMeans, X, unsupervised_learner.predictions[0]['KMeans'], plot_kwargs)

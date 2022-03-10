import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.datasets._samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

sns.set_context('poster')

sns.set_color_codes()

plot_kwargs = {'s': 10, 'linewidths': 0.1}

centers = [[2, 1], [-1.5, -1], [1, -1], [-2, 2]]
cluster_std = [0.1, 0.1, 0.1, 0.1]

X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=cluster_std, random_state=53)

plt.scatter(X[:, 0], X[:, 1], **plot_kwargs)
plt.show()

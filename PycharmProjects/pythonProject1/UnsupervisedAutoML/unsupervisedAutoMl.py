import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pprint import pprint
from sklearn.datasets._samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

# sns.set_context('poster')
#
# sns.set_color_codes()
#
# plot_kwargs = {'s': 10, 'linewidths': 0.1}
#
# centers = [[2, 1], [-1.5, -1], [1, -1], [-2, 2]]
# cluster_std = [0.4, 0.5, 0.6, 0.5]
#
# X, y = make_blobs(n_samples=1000, centers=centers, cluster_std=cluster_std, random_state=53)
#
# plt.scatter(X[:, 0], X[:, 1], **plot_kwargs)
# plt.show()


class Unsupervised_AutoMl:
    def __init__(self, estimators=None, transformers=None):
        self.estimators = estimators
        self.transformers = transformers
        pass

    def fit_predict(self, X, y=None):

        predictions = []
        performance_metrics = {}
        for estimator in self.estimators:
            labels = estimator['estimator'](*estimator['args'], **estimator['kwargs']).fit_predict(X)
            estimator['estimator'].n_clusters_ = len(np.unique(labels))
            metrics = self._get_cluster_metrics(estimator['estimator'].__name__, estimator['estimator'].n_clusters_, X,
                                                labels, y)
            predictions.append({estimator['estimator'].__name__: labels})
            performance_metrics[estimator['estimator'].__name__] = metrics
        self.predictions = predictions
        self.performance_metrics = performance_metrics
        return predictions, performance_metrics

    def _get_cluster_metrics(self, name, n_cluster_, X, pred_labels, true_labels=None):
        from sklearn.metrics import homogeneity_score, \
            completeness_score, \
            v_measure_score, \
            adjusted_rand_score, \
            adjusted_mutual_info_score, \
            silhouette_score

        print("""############## %s metrics ###########""" % name)
        if len(np.unique(pred_labels)) >= 2:
            silh_co = silhouette_score(X, pred_labels)

            if true_labels is not None:
                h_score = homogeneity_score(true_labels, pred_labels)
                c_score = completeness_score(true_labels, pred_labels)
                vm_score = v_measure_score(true_labels, pred_labels)
                adj_r_score = adjusted_rand_score(true_labels, pred_labels)
                adj_mut_score = adjusted_mutual_info_score(true_labels, pred_labels)
                metrics = {"Silhouette Coefficient": silh_co,
                           "Estimated number of cluster": n_cluster_,
                           "Homogeneity": h_score,
                           "Completeness": c_score,
                           "V-measure": vm_score,
                           "Adjusted Rand Index": adj_r_score,
                           "Adjusted Mutual Information": adj_mut_score}
                for k, v in metrics.items():
                    print("\t%s: %0.3f" % (k, v))
                return metrics
            metrics = {"Silhouette Coefficient": silh_co,
                       "Estimated number of cluster": n_cluster_}

            for k, v in metrics.items():
                print("\t%s: %0.3f" % (k, v))

            return metrics
        else:
            print("\t# of predicted labels is {}, can not produce metrics.\n".format(np.unique(pred_labels)))

    def plot_all_clusters(self, estimators, labels, X, plot_kwargs):
        fig = plt.figure()
        for i, algorithm in enumerate(labels):
            quotinent = np.divide(len(estimators), 2)

            if isinstance(quotinent, int):
                dim_1 = 2
                dim_2 = quotinent
            else:
                dim_1 = np.ceil(quotinent)
                dim_2 = 3
            palette = sns.color_palette('deep', np.unique(algorithm[estimators[i]['estimator'].__name__]).max() + 1)
            colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in algorithm[estimators[i]['estimator'].__name__]]
            # print(dim_1, dim_2)
            plt.subplot(int(dim_1 - 1), dim_2, i + 1)
            plt.scatter(X[:, 0], X[:, 1], c=colors, **plot_kwargs)
            plt.title('{}Cluster'.format(str(estimators[i]['estimator'].__name__)), fontsize=8)
        plt.show()

    def plot_clusters(self, estimators, X, labels, plot_kwargs):
        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

        plt.scatter(X[:, 0], X[:, 1], c=colors, **plot_kwargs)
        plt.title('{}Clusters'.format(str(estimators.__name__)), fontsize=14)
        plt.show()




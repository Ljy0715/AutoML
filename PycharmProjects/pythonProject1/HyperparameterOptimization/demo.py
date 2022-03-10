from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import load_iris, load_digits, load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import tree
from sklearn import gaussian_process
from sklearn.linear_model import PoissonRegressor, PassiveAggressiveClassifier


# # 产生随机分类数据集，10个特征， 2个类别
# X_train, y_train = make_classification(n_samples=1000, n_features=10, n_classes=2)

# digits
data_set = load_digits()
# print(X_train.shape, y_train.shape)
X_train, X_test, y_train, y_test = train_test_split(data_set.data, data_set.target, test_size=0.2, random_state=42)


# # hr dataset
# hr_data = pd.read_csv('data/hr.csv', header=0)
# hr_data.head()
# print(" Data Set Shape ", hr_data.shape)
# print(list(hr_data.columns))
# print(" Sample Data", hr_data.head())
#
# data_trnsf = pd.get_dummies(hr_data, columns=['salary', 'sales'])
# X = data_trnsf.drop('left', axis=1)
#
# X_train, X_test, y_train, y_test = train_test_split(X, data_trnsf.left, test_size=0.3, random_state=42)

# 先看看不调参的结果：
rf = RandomForestClassifier()
gp = gaussian_process.GaussianProcessClassifier()
gml = PassiveAggressiveClassifier()
print(np.mean(cross_val_score(gml, X_train, y_train, cv=20, scoring='accuracy')))
# print(np.mean(cross_val_score(gp, X_train, y_train, cv=20, scoring='accuracy')))
print(np.mean(cross_val_score(rf, X_train, y_train, cv=20, scoring='accuracy')))
# print(y_train)
# print(type(X_train), type(y_train))


# bayes rf
# def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
#     val = cross_val_score(
#         RandomForestClassifier(n_estimators=int(n_estimators),
#                                min_samples_split=int(min_samples_split),
#                                max_features=min(max_features, 0.999),  # float
#                                max_depth=int(max_depth),
#                                random_state=2),
#         X_train, y_train, scoring='accuracy', cv=5).mean()
#     return val
#
#
# # rf实例化一个bayes优化对象
# rf_bo = BayesianOptimization(
#         rf_cv,
#         {'n_estimators': (10, 250),
#          'min_samples_split': (2, 25),
#          'max_features': (0.1, 0.999),
#          'max_depth': (5, 15)})

# # 最优化
# rf_bo.maximize(n_iter=1)
#
# print(rf_bo.max)
# # print(rf_bo.max['target'])
# print(cross_val_score(
#         RandomForestClassifier(n_estimators=int(rf_bo.max['params']['n_estimators']),
#                                min_samples_split=int(rf_bo.max['params']['min_samples_split']),
#                                max_features=rf_bo.max['params']['max_features'],  # float
#                                max_depth=int(rf_bo.max['params']['max_depth']),
#                                random_state=2), X_train, y_train, scoring='accuracy', cv=5).mean())
#
# rf_best = RandomForestClassifier(n_estimators=int(rf_bo.max['params']['n_estimators']),
#                                  min_samples_split=int(rf_bo.max['params']['min_samples_split']),
#                                  max_features=rf_bo.max['params']['max_features'],  # float
#                                  max_depth=int(rf_bo.max['params']['max_depth']),
#                                  random_state=2)
# rf_best.fit(X_train, y_train)
# print(rf_best.score(X_test, y_test))

# bayes knn


# def knn_cv(n_neighbors, leaf_size):
#     val = cross_val_score(
#         KNeighborsClassifier(n_neighbors=int(n_neighbors),
#                              leaf_size=int(leaf_size),
#                              metric='euclidean'),
#         X_train, y_train, scoring='accuracy', cv=5).mean()
#     return val
#
#
# # knn实例化一个bayes优化对象
# knn_bo = BayesianOptimization(
#         knn_cv,
#         {'n_neighbors': (1, 20),
#          'leaf_size': (1, 50)})
# knn_bo.maximize(n_iter=1)
#
# print(knn_bo.max)
# # print(knn_bo.max['target'])
# print(cross_val_score(KNeighborsClassifier(n_neighbors=int(knn_bo.max['params']['n_neighbors']),
#                                            leaf_size=int(knn_bo.max['params']['leaf_size']),
#                                            metric='euclidean'), X_train, y_train, scoring='accuracy', cv=5).mean())
# knn_best = KNeighborsClassifier(n_neighbors=int(knn_bo.max['params']['n_neighbors']),
#                                 leaf_size=int(knn_bo.max['params']['leaf_size']),
#                                 metric='euclidean')
# knn_best.fit(X_train, y_train)
# print(knn_best.score(X_test, y_test))


# bayes svc
# def svc_cv(C, degree, coef0, tol, cache_size):
#     val = cross_val_score(
#         svm.SVC(C=min(C, 5),
#                 degree=int(degree),
#                 coef0=min(coef0, 0.999),
#                 tol=min(tol, 0.999),
#                 cache_size=int(cache_size)),
#         X_train, y_train, scoring='accuracy', cv=5).mean()
#     return val
#
#
# # svc实例化一个bayes优化对象
# svc_bo = BayesianOptimization(
#         svc_cv,
#         {'C': (0, 10),
#          'degree': (1, 10),
#          'coef0': (0, 0.999),
#          'tol': (0.001, 0.999),
#          'cache_size': (10, 500)})

# svc_bo.maximize(n_iter=1)
#
# print(svc_bo.max)
# # print(knn_bo.max['target'])
# print(cross_val_score(svm.SVC(C=svc_bo.max['params']['C'],
#                               degree=int(svc_bo.max['params']['degree']),
#                               coef0=svc_bo.max['params']['coef0'],
#                               tol=svc_bo.max['params']['tol'],
#                               cache_size=svc_bo.max['params']['cache_size']),
#                       X_train, y_train, scoring='accuracy', cv=5).mean())
#
# svc_best = svm.SVC(C=svc_bo.max['params']['C'],
#                    degree=int(svc_bo.max['params']['degree']),
#                    coef0=svc_bo.max['params']['coef0'],
#                    tol=svc_bo.max['params']['tol'],
#                    cache_size=svc_bo.max['params']['cache_size'])
# svc_best.fit(X_train, y_train)
# print(svc_best.score(X_test, y_test))


# # bayes decision tree
# def dt_cv(ccp_alpha, min_samples_split, max_features, max_depth):
#     val = cross_val_score(
#         RandomForestClassifier(ccp_alpha=int(ccp_alpha),
#                                min_samples_split=int(min_samples_split),
#                                max_features=min(max_features, 0.999),  # float
#                                max_depth=int(max_depth),
#                                random_state=2),
#         X_train, y_train, scoring='accuracy', cv=5).mean()
#     return val
#
#
# # rf实例化一个bayes优化对象
# dt_bo = BayesianOptimization(
#         dt_cv,
#         {'ccp_alpha': (0, 1),
#          'min_samples_split': (1, 25),
#          'max_features': (0.1, 0.999),
#          'max_depth': (0, 10)})
#
# # 最优化
# dt_bo.maximize(n_iter=1)
#
# print(dt_bo.max)
# # print(rf_bo.max['target'])
# print(cross_val_score(
#         tree.DecisionTreeClassifier(ccp_alpha=int(dt_bo.max['params']['ccp_alpha']),
#                                     min_samples_split=int(dt_bo.max['params']['min_samples_split']),
#                                     max_features=dt_bo.max['params']['max_features'],  # float
#                                     max_depth=int(dt_bo.max['params']['max_depth']),
#                                     random_state=2), X_train, y_train, scoring='accuracy', cv=5).mean())
#
# dt_best = tree.DecisionTreeClassifier(ccp_alpha=int(dt_bo.max['params']['ccp_alpha']),
#                                       min_samples_split=int(dt_bo.max['params']['min_samples_split']),
#                                       max_features=dt_bo.max['params']['max_features'],  # float
#                                       max_depth=int(dt_bo.max['params']['max_depth']),
#                                       random_state=2)
# dt_best.fit(X_train, y_train)
# print(dt_best.score(X_test, y_test))


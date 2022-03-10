import numpy as np
from sklearn.datasets import load_iris, load_digits, load_diabetes, load_wine, load_breast_cancer, load_boston,\
     fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn import svm
from sklearn import tree
from sklearn import gaussian_process
from sklearn.linear_model import PoissonRegressor, PassiveAggressiveRegressor
from time import time
from tensorflow import keras
from sklearn import metrics
from bayes_opt import BayesianOptimization
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler

# data_set = fetch_california_housing()
# X_train, X_test, y_train, y_test = train_test_split(data_set.data, data_set.target, test_size=0.2, random_state=42)
# X_train = X_train[: 500]
# y_train = y_train[: 500]
# X_test = X_test[500: 1000]
# y_test = y_test[500: 1000]

# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# X_train = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# y_train = raw_df.values[1::2, 2]


data = pd.read_excel('/home/ljy/scikit_learn_data/x_train_DOE.xlsx', header=None)
# myData = pd.DataFrame(data=data)
myData = data
# print(myData)
x = myData.reindex(columns=[0, 1, 2, 3, 4, 5, 6, 7])

y = myData.reindex(columns=[9])

x = np.array(x)
y = np.array(y)

scaler_x = StandardScaler()
scaler_x.fit(x)
x = scaler_x.transform(x)

# scaler_y = MinMaxScaler()
scaler_y = StandardScaler()
scaler_y.fit(y)
y = scaler_y.transform(y)

# print(y)
y = np.ravel(y)
X_train = x[:450]
X_test = x[450:500]
y_train = y[:450]
y_test = y[450:500]

print(X_train.shape, y_train.shape)
# print(type(X_train), type(y_train))
# print(X_train, y_train)

start = time()


# 1.svm
# bayes svc
def svc_cv(C, degree, coef0, tol, cache_size):
    val = cross_val_score(
        svm.SVR(C=min(C, 5),
                degree=int(degree),
                coef0=min(coef0, 0.999),
                tol=min(tol, 0.999),
                cache_size=int(cache_size)),
        X_train, y_train, scoring='r2', cv=5).mean()
    return val


# svc实例化一个bayes优化对象
svc_bo = BayesianOptimization(
        svc_cv,
        {'C': (0.01, 10),
         'degree': (1, 10),
         'coef0': (0, 0.999),
         'tol': (0.001, 0.999),
         'cache_size': (10, 500)})
print('svm optimize')
svc_bo.maximize(n_iter=25)
svr = svm.SVR(C=svc_bo.max['params']['C'],
              degree=int(svc_bo.max['params']['degree']),
              coef0=svc_bo.max['params']['coef0'],
              tol=svc_bo.max['params']['tol'],
              cache_size=svc_bo.max['params']['cache_size'])
svr.fit(X_train, y_train)
result_svr = svr.predict(X_train)
# print('y_true \n', y_train)
# print('result_svr \n', result_svr)


# 2.knn
# bayes knn
def knn_cv(n_neighbors, leaf_size):
    val = cross_val_score(
        KNeighborsRegressor(n_neighbors=int(n_neighbors),
                            leaf_size=int(leaf_size),
                            metric='euclidean'),
        X_train, y_train, scoring='r2', cv=5).mean()
    return val


knn_bo = BayesianOptimization(
        knn_cv,
        {'n_neighbors': (1, 20),
         'leaf_size': (1, 50)})
print('knn optimize')
knn_bo.maximize(n_iter=25)


# 3.DT
# bayes decision tree
def dt_cv(ccp_alpha, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        tree.DecisionTreeRegressor(ccp_alpha=int(ccp_alpha),
                                   min_samples_split=int(min_samples_split),
                                   max_features=min(max_features, 0.999),  # float
                                   max_depth=int(max_depth),
                                   random_state=2),
        X_train, y_train, scoring='r2', cv=5).mean()
    return val


# dt实例化一个bayes优化对象
dt_bo = BayesianOptimization(
        dt_cv,
        {'ccp_alpha': (0.001, 0.999),
         'min_samples_split': (2, 25),
         'max_features': (0.1, 0.999),
         'max_depth': (1, 10)})
print('dt optimize')
dt_bo.maximize(n_iter=25)


# 4.Random Forest
# bayes rf
def rf_rg(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestRegressor(n_estimators=int(n_estimators),
                              min_samples_split=int(min_samples_split),
                              max_features=min(max_features, 0.999),  # float
                              max_depth=int(max_depth),
                              random_state=2), X_train, y_train, scoring='r2').mean()
    # print(val)
    return val


rf_bo = BayesianOptimization(
        rf_rg,
        {'n_estimators': (10, 250),
         'min_samples_split': (2, 25),
         'max_features': (0.1, 0.999),
         'max_depth': (5, 15)})

print('rf optimize')
rf_bo.maximize(n_iter=25)


# 5. GradientBoostRegressor

def gb_rg(learning_rate, n_estimators, subsample, min_samples_split, min_samples_leaf, max_depth, alpha, tol):
    val = cross_val_score(
        GradientBoostingRegressor(learning_rate=learning_rate,
                                  n_estimators=int(n_estimators),
                                  subsample=subsample,
                                  min_samples_split=int(min_samples_split),
                                  min_samples_leaf=int(min_samples_leaf),
                                  max_depth=int(max_depth),
                                  alpha=alpha,
                                  tol=tol), X_train, y_train, scoring='r2').mean()
    # print(val)
    return val


gb_bo = BayesianOptimization(
        gb_rg,
        {'learning_rate': (0.0001, 0.02),
         'n_estimators': (50, 1000),
         'subsample': (0.001, 1),
         'min_samples_split': (2, 10),
         'min_samples_leaf': (2, 10),
         'max_depth': (2, 10),
         'alpha': (0.01, 0.999),
         'tol': (0.1, 0.999)})

print('gb optimize')
gb_bo.maximize(n_iter=25)


# 6. kriging
def gp_cv(n_restarts_optimizer, alpha):
    val = cross_val_score(
        gaussian_process.GaussianProcessRegressor(n_restarts_optimizer=int(n_restarts_optimizer),
                                                  alpha=alpha,
                                                  random_state=2),
        X_train, y_train, scoring='r2', cv=5).mean()
    return val


gp_bo = BayesianOptimization(
        gp_cv,
        {'n_restarts_optimizer': (0, 0.000001),
         'alpha': (1e-10, 1.1e-10)})
print('gp optimize')
gp_bo.maximize(n_iter=25)


# 7. PassiveAggressiveClassifier
# PassiveAggressiveClassifier


def pa_cv(C, max_iter, tol, validation_fraction, n_iter_no_change):
    val = cross_val_score(
        PassiveAggressiveRegressor(C=C,
                                   max_iter=int(max_iter),
                                   tol=tol,
                                   validation_fraction=validation_fraction,
                                   n_iter_no_change=int(n_iter_no_change)),
        X_train, y_train, scoring='r2', cv=5).mean()
    return val


pa_bo = BayesianOptimization(
        pa_cv,
        {'C': (0.001, 100),
         'max_iter': (500, 3000),
         'tol': (0.00001, 0.1),
         'validation_fraction': (0.001, 0.999),
         'n_iter_no_change': (1, 10)})
print('pa optimize')
pa_bo.maximize(n_iter=10)


# 8.ET
# bayes decision tree
def et_cv(ccp_alpha, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        tree.ExtraTreeRegressor(ccp_alpha=int(ccp_alpha),
                                min_samples_split=int(min_samples_split),
                                max_depth=int(max_depth),
                                max_features=min(max_features, 0.999),
                                random_state=2),
        X_train, y_train, scoring='r2', cv=5).mean()
    return val


# dt实例化一个bayes优化对象
et_bo = BayesianOptimization(
        et_cv,
        {'ccp_alpha': (0, 1),
         'min_samples_split': (2, 25),
         'max_features': (0.01, 0.999),
         'max_depth': (1, 10)})
print('dt optimize')
et_bo.maximize(n_iter=25)

pipe_svm = Pipeline([
                     ('svm', svm.SVR(C=svc_bo.max['params']['C'],
                                     degree=int(svc_bo.max['params']['degree']),
                                     coef0=svc_bo.max['params']['coef0'],
                                     tol=svc_bo.max['params']['tol'],
                                     cache_size=svc_bo.max['params']['cache_size']))])

pipe_knn = Pipeline([
                     ('knn', KNeighborsRegressor(n_neighbors=int(knn_bo.max['params']['n_neighbors']),
                                                 leaf_size=int(knn_bo.max['params']['leaf_size']),
                                                 metric='euclidean'))])

pipe_dt = Pipeline([

                    ('dt', tree.DecisionTreeRegressor(ccp_alpha=int(dt_bo.max['params']['ccp_alpha']),
                                                      min_samples_split=int(dt_bo.max['params']['min_samples_split']),
                                                      max_features=dt_bo.max['params']['max_features'],  # float
                                                      max_depth=int(dt_bo.max['params']['max_depth']),
                                                      random_state=2))])

pipe_rf = Pipeline([
                    ('rf', RandomForestRegressor(n_estimators=int(rf_bo.max['params']['n_estimators']),
                                                 min_samples_split=int(rf_bo.max['params']['min_samples_split']),
                                                 max_features=rf_bo.max['params']['max_features'],  # float
                                                 max_depth=int(rf_bo.max['params']['max_depth']),
                                                 random_state=42))])

pipe_gb = Pipeline([
                    ('gb', GradientBoostingRegressor(learning_rate=gb_bo.max['params']['learning_rate'],
                                                     n_estimators=int(gb_bo.max['params']['n_estimators']),
                                                     subsample=gb_bo.max['params']['subsample'],
                                                     min_samples_split=int(gb_bo.max['params']['min_samples_split']),
                                                     min_samples_leaf=int(gb_bo.max['params']['min_samples_leaf']),
                                                     max_depth=int(gb_bo.max['params']['max_depth']),
                                                     alpha=gb_bo.max['params']['alpha'],
                                                     tol=gb_bo.max['params']['tol']))])


pipe_gp = Pipeline([
                    ('gp', gaussian_process.GaussianProcessRegressor(
                        n_restarts_optimizer=int(gp_bo.max['params']['n_restarts_optimizer']),
                        alpha=int(gp_bo.max['params']['alpha']),
                        random_state=42))])

pipe_pa = Pipeline([
                    ('PassiveAggressiveClassifier', PassiveAggressiveRegressor(C=pa_bo.max['params']['C'],
                                                                               max_iter=int(pa_bo.max['params']['max_iter']),
                                                                               tol=pa_bo.max['params']['tol'],
                                                                               validation_fraction=pa_bo.max['params']['validation_fraction'],
                                                                               n_iter_no_change=int(pa_bo.max['params']['n_iter_no_change'])))])

pipe_et = tree.ExtraTreeRegressor(ccp_alpha=int(et_bo.max['params']['ccp_alpha']),
                                  min_samples_split=int(et_bo.max['params']['min_samples_split']),
                                  max_features=dt_bo.max['params']['max_features'],
                                  max_depth=int(et_bo.max['params']['max_depth']),
                                  random_state=2)

pipe_dic = {0: 'Support Vector Machines', 1: 'K Nearset Neighbours', 2: 'Decision Tree', 3: 'Random Forest',
            4: 'gradient boosting', 5: 'gauss process', 6: 'PassiveAggressiveClassifier', 7: 'ExtraTree'}

pipelines = [pipe_svm, pipe_knn, pipe_dt, pipe_rf, pipe_gb, pipe_gp, pipe_pa, pipe_et]
for pipe in pipelines:
    pipe.fit(X_train, y_train)

end = time()
print('in mnist spend %.4f' % (end-start))
best_accuracy = 0
best_regressor = 0
best_pipeline = ''
for idx, val in enumerate(pipelines):
    print('%s pipeline test accuracy: %.3f' % (pipe_dic[idx], val.score(X_test, y_test)))
    if np.abs(val.score(X_test, y_test)) > abs(best_accuracy):
        best_accuracy = val.score(X_test, y_test)
        best_pipeline = val
        best_regressor = idx
print('%s Classifier has the best accuracy of %.4f' % (pipe_dic[best_regressor], best_accuracy))

# pa_best = PassiveAggressiveRegressor(C=pa_bo.max['params']['C'],
#                                      max_iter=int(pa_bo.max['params']['max_iter']),
#                                      tol=pa_bo.max['params']['tol'],
#                                      validation_fraction=pa_bo.max['params']['validation_fraction'],
#                                      n_iter_no_change=int(pa_bo.max['params']['n_iter_no_change']))
# pa_best.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

y_test = y_test.reshape((50, 1))
y_pred = y_pred.reshape((50, 1))

y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

y_pred = np.ravel(y_pred)
y_test = np.ravel(y_test)

error_sum = 0
for i in range(50):
    print(y_pred[i], y_test[i])
    # error_sum = error_sum + abs(y_pred[i] - y_test[i])/(y_test[i])
# print(error_sum/50)
print("mae score", metrics.mean_absolute_error(y_test, y_pred))
print("mse score", metrics.mean_squared_error(y_test, y_pred))
# print(y_test.shape, y_pred.shape)
print("r2 score", metrics.r2_score(y_test, y_pred), '\n')

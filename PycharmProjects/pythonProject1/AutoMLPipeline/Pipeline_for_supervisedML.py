import numpy as np
from sklearn.datasets import load_iris, load_digits, load_diabetes, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn import gaussian_process
from sklearn.linear_model import PoissonRegressor, PassiveAggressiveClassifier
from time import time
from tensorflow import keras
from multiprocessing import Pool
from bayes_opt import BayesianOptimization


def CunstomLog(x):
    return np. log(x)


def PreprocData(X, Y):
    pipe = make_pipeline(FunctionTransformer(CunstomLog), StandardScaler())
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    pipe.fit(X_train, y_train)
    return pipe.transform(X_test), y_test


# data_set = load_diabetes()
# data_set = load_iris()
# data_set = load_wine()
data_set = load_breast_cancer()
# data_set = load_digits()
# print(X)
# X_transformed, Y_transformed = PreprocData(X, Y)
# print(X_transformed)

# X_train, X_test, y_train, y_test = train_test_split(data_set.data, data_set.target, test_size=0.2, random_state=42)

(X, y), (X_test, y_test) = keras.datasets.mnist.load_data()

num_train = 100
num_test = 100

# 创建train样本
X_train = X[0:num_train]
y_train = y[0:num_train]
X_train = X_train.reshape(num_train, 784)
X_train = np.array(X_train)
y_train = np.array(y_train)

# 创建test样本
X_test = X_test[0:num_train]
y_test = y_test[0:num_train]
X_test = X_test.reshape(num_train, 784)
X_test = np.array(X_test)
y_test = np.array(y_test)



# 1.svm
# bayes svc
def svc_cv(C, degree, coef0, tol, cache_size):
    val = cross_val_score(
        svm.SVC(C=min(C, 5),
                degree=int(degree),
                coef0=min(coef0, 0.999),
                tol=min(tol, 0.999),
                cache_size=int(cache_size)),
        X_train, y_train, scoring='accuracy', cv=5).mean()
    return val


# 2.knn
# bayes knn
def knn_cv(n_neighbors, leaf_size):
    val = cross_val_score(
        KNeighborsClassifier(n_neighbors=int(n_neighbors),
                             leaf_size=int(leaf_size),
                             metric='euclidean'),
        X_train, y_train, scoring='accuracy', cv=5).mean()
    return val


# 3.DT
# bayes decision tree
def dt_cv(ccp_alpha, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        tree.DecisionTreeClassifier(ccp_alpha=int(ccp_alpha),
                                    min_samples_split=int(min_samples_split),
                                    max_features=min(max_features, 0.999),  # float
                                    max_depth=int(max_depth),
                                    random_state=2),
        X_train, y_train, scoring='accuracy', cv=5).mean()
    return val

# 4.Random Forest


# bayes rf
def rf_cv(n_estimators, min_samples_split, max_features, max_depth):
    val = cross_val_score(
        RandomForestClassifier(n_estimators=int(n_estimators),
                               min_samples_split=int(min_samples_split),
                               max_features=min(max_features, 0.999),  # float
                               max_depth=int(max_depth),
                               random_state=2),
        X_train, y_train, scoring='accuracy', cv=5).mean()
    return val


# 8. kriging
def gp_cv(n_restarts_optimizer, max_iter_predict):
    val = cross_val_score(
        gaussian_process.GaussianProcessClassifier(n_restarts_optimizer=int(n_restarts_optimizer),
                                                   max_iter_predict=int(max_iter_predict),
                                                   random_state=2),
        X_train, y_train, scoring='accuracy', cv=5).mean()
    return val

# 9. rbf
# PassiveAggressiveClassifier


def rbf_cv(C, max_iter, tol, validation_fraction, n_iter_no_change):
    val = cross_val_score(
        PassiveAggressiveClassifier(C=C,
                                    max_iter=int(max_iter),
                                    tol=tol,
                                    validation_fraction=validation_fraction,
                                    n_iter_no_change=int(n_iter_no_change)),
        X_train, y_train, scoring='accuracy', cv=5).mean()
    return val


start = time()
# svc实例化一个bayes优化对象
svc_bo = BayesianOptimization(
        svc_cv,
        {'C': (0.01, 10),
         'degree': (1, 10),
         'coef0': (0, 0.999),
         'tol': (0.001, 0.999),
         'cache_size': (10, 500)})

knn_bo = BayesianOptimization(
        knn_cv,
        {'n_neighbors': (1, 20),
         'leaf_size': (1, 50)})


# dt实例化一个bayes优化对象
dt_bo = BayesianOptimization(
        dt_cv,
        {'ccp_alpha': (0, 1),
         'min_samples_split': (2, 25),
         'max_features': (0.1, 0.999),
         'max_depth': (1, 10)})

rf_bo = BayesianOptimization(
        rf_cv,
        {'n_estimators': (10, 250),
         'min_samples_split': (2, 25),
         'max_features': (0.1, 0.999),
         'max_depth': (5, 15)})


gp_bo = BayesianOptimization(
        gp_cv,
        {'n_restarts_optimizer': (0, 20),
         'max_iter_predict': (10, 400)})

rbf_bo = BayesianOptimization(
        rbf_cv,
        {'C': (0.001, 100),
         'max_iter': (500, 2000),
         'tol': (0.00001, 0.1),
         'validation_fraction': (0.001, 0.999),
         'n_iter_no_change': (1, 10)})


def countdown(n):
    while n > 0:
        n -= 1


def start_optimize(i):
    if i == 0:
        print('1.svc Optimize')
        svc_bo.maximize(n_iter=45)
        return svc_bo
    elif i == 1:
        print('2.knn Optimize')
        knn_bo.maximize(n_iter=5)
    elif i == 2:
        print('3. decision tree Optimize')
        dt_bo.maximize(n_iter=5)
    elif i == 3:
        print('4.random forest Optimize')
        rf_bo.maximize(n_iter=1)
    elif i == 4:
        print('5.gauss process Optimize')
        gp_bo.maximize(n_iter=1)
    elif i == 5:
        print('6.rbf Optimize')
        rbf_bo.maximize(n_iter=5)


if __name__ == "__main__":
    count = 2e7
    start = time()
    # n_processes = os.cpu_count()
    n_processes = 6  # 进程数
    pool = Pool(processes=n_processes)  # 进程池
    for i in range(n_processes):
        pool.apply_async(start_optimize, (i,))  # 启动多进程

    pool.close()  # 使进程池不能添加新任务
    pool.join()  # 等待进程结束
    print(time() - start)

pipe_svm = Pipeline([('ss1', StandardScaler()),
                     ('svm', svm.SVC(C=svc_bo.max['params']['C'],
                                     degree=int(svc_bo.max['params']['degree']),
                                     coef0=svc_bo.max['params']['coef0'],
                                     tol=svc_bo.max['params']['tol'],
                                     cache_size=svc_bo.max['params']['cache_size']))])

pipe_knn = Pipeline([('ss2', StandardScaler()),
                     ('knn', KNeighborsClassifier(n_neighbors=int(knn_bo.max['params']['n_neighbors']),
                                                  leaf_size=int(knn_bo.max['params']['leaf_size']),
                                                  metric='euclidean'))])

pipe_dt = Pipeline([('ss3', StandardScaler()),
                    ('minmax', MinMaxScaler()),
                    ('dt', tree.DecisionTreeClassifier(ccp_alpha=int(dt_bo.max['params']['ccp_alpha']),
                                                       min_samples_split=int(dt_bo.max['params']['min_samples_split']),
                                                       max_features=dt_bo.max['params']['max_features'],  # float
                                                       max_depth=int(dt_bo.max['params']['max_depth']),
                                                       random_state=2))])

pipe_rf = Pipeline([('ss4', StandardScaler()),
                    ('rf', RandomForestClassifier(n_estimators=int(rf_bo.max['params']['n_estimators']),
                                                  min_samples_split=int(rf_bo.max['params']['min_samples_split']),
                                                  max_features=rf_bo.max['params']['max_features'],  # float
                                                  max_depth=int(rf_bo.max['params']['max_depth']),
                                                  random_state=42))])


# 5. logic regression

# 6. boosting GBMS

# 7. linear regression

pipe_gp = Pipeline([('ss4', StandardScaler()),
                    ('gp', gaussian_process.GaussianProcessClassifier(
                        n_restarts_optimizer=int(gp_bo.max['params']['n_restarts_optimizer']),
                        max_iter_predict=int(gp_bo.max['params']['max_iter_predict']),
                        random_state=42))])

pipe_rbf = Pipeline([('ss4', StandardScaler()),
                    ('PassiveAggressiveClassifier', PassiveAggressiveClassifier(C=rbf_bo.max['params']['C'],
                                                                                max_iter=int(rbf_bo.max['params']['max_iter']),
                                                                                tol=rbf_bo.max['params']['tol'],
                                                                                validation_fraction=rbf_bo.max['params']['validation_fraction'],
                                                                                n_iter_no_change=int(rbf_bo.max['params']['n_iter_no_change'])))])

pipe_dic = {0: 'K Nearset Neighbours', 1: 'Decision Tree', 2: 'Random Forest', 3: 'Support Vector Machines',
            4: 'gauss process', 5: 'PassiveAggressiveClassifier'}

pipelines = [pipe_knn, pipe_dt, pipe_rf, pipe_svm, pipe_gp, pipe_rbf]
for pipe in pipelines:
    pipe.fit(X_train, y_train)

end = time()
print('in mnist spend %.4f' % (end-start))
best_accuracy = 0
best_classifier = 0
best_pipeline = ''
for idx, val in enumerate(pipelines):
    print('%s pipeline test accuracy: %.3f' % (pipe_dic[idx], val.score(X_test, y_test)))
    if val.score(X_test, y_test) > best_accuracy:
        best_accuracy = val.score(X_test, y_test)
        best_pipeline = val
        best_classifier = idx
print('%s Classifier has the best accuracy of %.4f' % (pipe_dic[best_classifier], best_accuracy))

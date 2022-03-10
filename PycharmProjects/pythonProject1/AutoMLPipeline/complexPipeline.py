import numpy as np
from sklearn.datasets import load_iris, load_digits, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree


def CunstomLog(x):
    return np. log(x)


def PreprocData(X, Y):
    pipe = make_pipeline(FunctionTransformer(CunstomLog), StandardScaler())
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    pipe.fit(X_train, y_train)
    return pipe.transform(X_test), y_test


# data_set = load_diabetes()
# data_set = load_iris()
data_set = load_digits()
# X, Y = iris.data, iris.target
# print(X)
# X_transformed, Y_transformed = PreprocData(X, Y)
# print(X_transformed)
X_train, X_test, y_train, y_test = train_test_split(data_set.data, data_set.target, test_size=0.2, random_state=42)

# svm
pipe_svm = Pipeline([('ss1', StandardScaler()), ('pca', PCA(n_components=2)), ('svm', svm.SVC(random_state=42))])

# knn
pipe_knn = Pipeline([('ss2', StandardScaler()), ('knn', KNeighborsClassifier(n_neighbors=6, metric='euclidean'))])

# Construct DT
pipe_dt = Pipeline([('ss3', StandardScaler()), ('minmax', MinMaxScaler()),
                    ('dt', tree.DecisionTreeClassifier(random_state=42))])

# Random Forest
num_trees = 100
max_features = 1
pipe_rf = Pipeline([('ss4', StandardScaler()), ('pca', PCA(n_components=2)),
                    ('rf', RandomForestClassifier(n_estimators=num_trees, max_features=max_features))])

pipe_dic = {0: 'K Nearset Neighbours', 1: 'Decision Tree', 2: 'Random Forest', 3: 'Support Vector Machines'}

pipelines = [pipe_knn, pipe_dt, pipe_rf, pipe_svm]
for pipe in pipelines:
    pipe.fit(X_train, y_train)

best_accuracy = 0
best_classifier = 0
best_pipeline = ''
for idx, val in enumerate(pipelines):
    print('%s pipeline test accuracy: %.3f' % (pipe_dic[idx], val.score(X_test, y_test)))
    if val.score(X_test, y_test) > best_accuracy:
        best_accuracy = val.score(X_test, y_test)
        best_pipeline = val
        best_classifier = idx
        print('%s Classifier has the best accuracy of %.2f' % (pipe_dic[best_classifier], best_accuracy))


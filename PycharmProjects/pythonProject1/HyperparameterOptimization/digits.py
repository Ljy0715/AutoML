from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import load_iris, load_digits, load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd


# # 产生随机分类数据集，10个特征， 2个类别
# X_train, y_train = make_classification(n_samples=1000, n_features=10, n_classes=2)

# digits
data_set = load_digits()
# data_set = load_iris()
# print(X_train.shape, y_train.shape)
X_train, X_test, y_train, y_test = train_test_split(data_set.data, data_set.target, test_size=0.2, random_state=42)

num_trees = 100
max_features = 3
attrition_forest = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
attrition_forest.fit(X_train, y_train)
print(attrition_forest.score(X_train, y_train))

rf = RandomForestClassifier()
print(np.mean(cross_val_score(rf, X_train, y_train, cv=20, scoring='accuracy')))

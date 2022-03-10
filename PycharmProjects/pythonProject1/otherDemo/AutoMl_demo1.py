import autosklearn.classification
from sklearn import datasets
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import keras.datasets.mnist
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
import autosklearn.regression
from autosklearn.experimental.askl2 import AutoSklearn2Classifier

# (X, y), (X_test, y_test) = keras.datasets.mnist.load_data()
#
# num_train = 1000
# num_test = 100
#
# # 创建train样本
# X_train = X[0:num_train]
# y_train = y[0:num_train]
# X_train = X_train.reshape(num_train, 784)
# X_train = np.array(X_train)
# y_train = np.array(y_train)
#
# # 创建test样本
# X_test = X_test[0:num_train]
# y_test = y_test[0:num_train]
# X_test = X_test.reshape(num_train, 784)
# X_test = np.array(X_test)
# y_test = np.array(y_test)


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

np.random.seed(116)
np.random.shuffle(x)
# 将x打乱
np.random.seed(116)
np.random.shuffle(y)

# print(y)
y = np.ravel(y)
X_train = x[:450]
X_test = x[450:500]
y_train = y[:450]
y_test = y[450:500]


# X, y = datasets.load_digits(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

automl = autosklearn.regression.AutoSklearnRegressor(time_left_for_this_task=600, per_run_time_limit=50)
automl.fit(X_train, y_train)

y_pred = automl.predict(X_test)
print(automl.show_models())

y_test = y_test.reshape((50, 1))
y_pred = y_pred.reshape((50, 1))

y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

y_pred = np.ravel(y_pred)
y_test = np.ravel(y_test)

for i in range(50):
    print(y_pred[i], y_test[i])

print("mae score", metrics.mean_absolute_error(y_test, y_pred))
print("mse score", metrics.mean_squared_error(y_test, y_pred))
# print(y_test.shape, y_pred.shape)
print("r2 score", metrics.r2_score(y_test, y_pred), '\n')



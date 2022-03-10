import numpy as np
import pandas as pd
import autokeras as ak
from sklearn.preprocessing import FunctionTransformer, StandardScaler, MinMaxScaler
from sklearn import metrics
from time import time


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

# X_train = tf.cast(X_train, tf.float64)
# y_train = tf.cast(y_train, tf.float64)

start = time()

reg = ak.StructuredDataRegressor(overwrite=True, max_trials=3)
reg.fit(X_train, y_train, epochs=500)

end = time()

y_pred = reg.predict(X_test)

y_test = y_test.reshape((50, 1))
y_pred = y_pred.reshape((50, 1))

y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

y_pred = np.ravel(y_pred)
y_test = np.ravel(y_test)

for i in range(50):
    print(y_pred[i], y_test[i])

print("spend:", end-start)
print("mae score", metrics.mean_absolute_error(y_test, y_pred))
print("mse score", metrics.mean_squared_error(y_test, y_pred))
# print(y_test.shape, y_pred.shape)
print("r2 score", metrics.r2_score(y_test, y_pred), '\n')

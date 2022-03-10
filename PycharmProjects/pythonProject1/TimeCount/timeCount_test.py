from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import timeCount
from sklearn.linear_model import LogisticRegression


nT = 100000000
nF = 10
nE = int(nT / nF)

X = np.random.rand(nT).reshape(nE, nF)

y = np.random.randint(2, size=nE)

scoring_data = np.random.rand(nF).reshape(1, -1)

knn = KNeighborsClassifier(11, algorithm='brute')

with timeCount.timer():
    knn.fit(X, y)

with timeCount.timer():
    knn.predict(scoring_data)

log_res = LogisticRegression(C=1e5)

with timeCount.timer():
    log_res.fit(X, y)

with timeCount.timer():
    prediction = log_res.predict(scoring_data)


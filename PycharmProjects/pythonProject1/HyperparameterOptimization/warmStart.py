from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from time import time

log_reg = LogisticRegression(C=10, tol=0.00001, max_iter=3000)
iris = datasets.load_iris()
X = iris.data
Y = iris.target

start = time()
log_reg.fit(X, Y)
end = time()

print("Time: {}".format(end-start))

log_reg.set_params(C=20)

start = time()
log_reg.fit(X, Y)
end = time()

print("Time: {}".format(end-start))

log_reg_warm = LogisticRegression(C=10, tol=0.00001, max_iter=100000, solver='sag', warm_start=True)
start = time()
log_reg_warm.fit(X, Y)
end = time()

print("Time: {}".format(end-start))

log_reg_warm.set_params(C=20)

start = time()
log_reg_warm.fit(X, Y)
end = time()

print("Time: {}".format(end-start))

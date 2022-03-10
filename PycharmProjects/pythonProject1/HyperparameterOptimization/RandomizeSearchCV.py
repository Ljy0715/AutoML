from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint
from sklearn.linear_model import LogisticRegression
from sklearn import datasets

param_grid = {'C': sp_randint(1, 100), 'penalty': ['l2']}

iris = datasets.load_iris()
X = iris.data
Y = iris.target

n_iter_search = 20
n_folds = 5
log_reg = LogisticRegression(max_iter=3000)
estimator = RandomizedSearchCV(log_reg, param_distributions=param_grid, n_iter=n_iter_search, cv=n_folds)

estimator.fit(X, Y)

print(estimator.best_score_)
print(estimator.best_params_)
print(estimator.get_params())
# print(estimator.cv_results_)

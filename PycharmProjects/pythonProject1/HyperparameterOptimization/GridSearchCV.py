from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import datasets
import pandas as pd

log_reg = LogisticRegression(max_iter=3000)
para_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 20, 50, 100], 'penalty': ['l2']}

n_folds = 5
estimator = GridSearchCV(log_reg, para_grid, cv=n_folds)
iris = datasets.load_iris()
X = iris.data
Y = iris.target

print(estimator.fit(X, Y))

# print(dir(estimator.best_estimator_))
print(estimator.best_score_)
print(estimator.best_params_)
# print(estimator.cv_results_)
# print(estimator.best_index_)
print(estimator.get_params())

df = pd.DataFrame(estimator.cv_results_)
df.head()
print(df)
print(df.head())
print(df[df['mean_test_score'] == df['mean_test_score'].max()])


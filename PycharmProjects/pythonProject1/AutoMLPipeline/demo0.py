from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb

classifier = xgb.XGBClassifier()
print(classifier)

iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
# print(X_train)
pipe_lr = Pipeline([('minmax', MinMaxScaler()), ('lr', LogisticRegression(random_state=42))])
pipe_lr.fit(X_train, y_train)

score = pipe_lr.score(X_train, y_train)
print('Logistic Regression pipeline test accuracy: %.3f' % score)


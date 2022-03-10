import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split

hr_data = pd.read_csv('data/hr.csv', header=0)
hr_data.head()
print(" Data Set Shape ", hr_data.shape)
print(list(hr_data.columns))
print(" Sample Data", hr_data.head())

data_trnsf = pd.get_dummies(hr_data, columns=['salary', 'sales'])
data_trnsf.columns
X = data_trnsf.drop('left', axis=1)
X.columns

X_train, X_test, Y_train, Y_test = train_test_split(X, data_trnsf.left, test_size=0.3, random_state=42)
print(X_train)

num_trees = 100
max_features = 3
attrition_gradientboost = GradientBoostingClassifier(n_estimators=num_trees, max_features=max_features)
attrition_gradientboost.fit(X_train, Y_train)

Y_pred = attrition_gradientboost.predict(X_test)
confusionMatrix = confusion_matrix(Y_test, Y_pred)    
print(confusionMatrix)

print('Accuracy of Decision Tree classifier on test set:{:.2f}', format(attrition_gradientboost.score(X_test, Y_test)))
print(classification_report(Y_test, Y_pred))

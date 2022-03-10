import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

hr_data = pd.read_csv('data/hr.csv', header=0)
hr_data.head()
print(" Data Set Shape ", hr_data.shape)
print(list(hr_data.columns))
print(" Sample Data", hr_data.head())

data_trnsf = pd.get_dummies(hr_data, columns=['salary', 'sales'])
# data_trnsf.columns
X = data_trnsf.drop('left', axis=1)
# X.columns

X_train, X_test, Y_train, Y_test = train_test_split(X, data_trnsf.left, test_size=0.3, random_state=42)
print(X_train)

attrition_classifier = LogisticRegression()
attrition_classifier.fit(X_train, Y_train)

Y_pred = attrition_classifier.predict(X_test)
confusionMatrix = confusion_matrix(Y_test, Y_pred)
print(confusionMatrix)

print('Accuracy of Decision Tree classifier on test set:{:.2f}',format(attrition_classifier.score(X_test, Y_test)))
print(classification_report(Y_test, Y_pred))

fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='AUC = %0.2f' % auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel(['False Positive Rate(FPR)'])
plt.ylabel(['True Positive Rate(TPR)'])
plt.title('Receiver Operating Characteristic(ROC) Cure')
plt.legend(loc="right")
plt.show()

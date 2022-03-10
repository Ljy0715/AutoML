import wget
from sklearn.datasets import fetch_kddcup99
from pprint import pprint
import numpy as np
import pandas as pd
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.regression import AutoSklearnRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
# from sklearn import preprocessing

# link_to_data = 'https://apsportal.ibm.com/exchange-api/v1/entries/8044492073eb964f46597b4be06ff5ea/data?accessKey=9561295fa407698694b1e254d0099600'
# filename = wget.download(link_to_data)
#
# df = pd.read_csv('GoSales_Tx_NaiveBays.csv')
# df.head()
#
# df = df.apply(LabelEncoder().fit_transform())
# df.head()


def encode_target_variable(df=None, target_column=None, y=None):
    if df is not None:
        df_type = isinstance(df, pd.core.frame.DataFrame)
        if df_type:
            if not np.issubdtype(df[target_column].dtype, np.number):
                le = LabelEncoder()
                df[target_column] = le.fit_transform(df[target_column])
                return df[target_column]
            return df[target_column]
        else:
            if not np.issubdtype(y.dtype, np.number):
                le = LabelEncoder()
                y = le.fit_transform(y)
                return y
            return y


def supervised_learner(type, X_train, y_train, X_test, y_test):
    if type == 'regression':
        automl = AutoSklearnRegressor(time_left_for_this_task=720, per_run_time_limit=72)
    else:
        automl = AutoSklearnClassifier(time_left_for_this_task=720, per_run_time_limit=72)
    automl.fit(X_train, y_train)

    y_hat = automl.predict(X_test)

    metric = accuracy_score(y_test, y_hat)

    return automl, y_hat, metric


def supervised_automl(data, target_column=None, type=None, y=None):
    df_type = isinstance(data, pd.core.frame.DataFrame)
    if df_type:
        data[target_column] = encode_target_variable(data, target_column)
        X_train, X_test, y_train, y_test = train_test_split(data.loc[:, data.columns != target_column],
                                                            data[target_column], random_state=1)
    else:
        y_encoded = encode_target_variable(y=y)
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, random_state=1)
    if type != None:
        automl, y_hat, metric = supervised_learner(type, X_train, y_train, X_test, y_test)
    elif type == 'regression':
        print("""There are more than 10 uniques numerical values in target column.
        Treating it as regression problem.""")
        automl, y_hat, metric = supervised_learner('regression', X_train, y_train, X_test, y_test)
    else:
        automl, y_hat, metric = supervised_learner('classification', X_train, y_train, X_test, y_test)
    return automl, y_hat, metric


# autoML, y_hat, metric = supervised_automl(df, target_column='PRODUCT_LINE')
dataset = fetch_kddcup99(subset='http', shuffle=True, percent10=True)

X = dataset.data
y = dataset.target

print(X.shape)
print(y.shape)

pprint(np.unique(y))

autoML, y_hat, metric = supervised_automl(X, y=y, type='classification')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

hr_data = pd.read_csv('data/hr.csv', header=0)
hr_data.head()
print(" Data Set Shape ", hr_data.shape)
print(list(hr_data.columns))
print(" Sample Data", hr_data.head())
hr_data[hr_data.dtypes[(hr_data.dtypes == "float64") | (hr_data.dtypes == "int64")].index.values].hist(figsize=[11, 11])
# plt.show()

scaler = StandardScaler()
hr_data_scaler = scaler.fit_transform(hr_data[['satisfaction_level']])
hr_data_scaler_df = pd.DataFrame(hr_data_scaler)
hr_data_scaler_df.max()
hr_data_scaler_df[hr_data_scaler_df.dtypes[(hr_data_scaler_df.dtypes == "float64")
                                           | (hr_data_scaler_df.dtypes == "int64")].index.values].hist(figsize=[11, 11])


minmax = MinMaxScaler()
hr_data_minmax = minmax.fit_transform(hr_data[['average_montly_hours', 'last_evaluation', 'number_project',
                                               'satisfaction_level']])
hr_data_minmax_df = pd.DataFrame(hr_data_minmax)
hr_data_minmax_df.max()
hr_data_minmax_df.min()
hr_data_minmax_df[hr_data_minmax_df.dtypes[(hr_data_minmax_df.dtypes == "float64")
                                           | (hr_data_minmax_df.dtypes == "int64")].index.values].hist(figsize=[11, 11])
plt.show()

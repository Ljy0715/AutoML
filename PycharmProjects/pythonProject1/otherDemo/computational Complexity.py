import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn-whitegrid')

n = 10
x = np.arange(1, n)
print(x)
df = pd.DataFrame({'x': x,
                   'O(1)': 0,
                   'O(n)': x,
                   'O(log_n)': np.log(x),
                  'O(n_log_n)': n * np.log(x),
                   'O(n2)': np.power(x, 2),
                   'O(n3)': np.power(x, 3)})
print(df)
labels = ['$O(1) - Constant$',
          '$O(\log{}n) - Logarithmic$',
          '$O(n) - Linear$',
          'SO(n^2) - Quadratics$',
          '$O(n^3) - Cubics$',
          'SO(n\log{}n) - N log n$']

for i, col in enumerate(df.columns.drop('x')):
    print(labels[i], col)
    plt.plot(df[col], label=labels[i])
plt.legend()
plt.ylim(0, 50)
plt.show()

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


def branin(x):
    x1 = x[0]
    x2 = x[1]
    a = 1
    b = 5.1 / (4 * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1 / (8 * np.pi)

    p1 = a * (x2 - (b * x1 ** 2) + (c * x1) - r) ** 2
    p2 = s * (1 - t) * np.cos(x1)
    p3 = s

    ret = p1 + p2 + p3

    return ret


# x = [5.6, 3.2]
# res = minimize(branin, x)
# print(res)
#
# log_res = LogisticRegression()
# print(log_res.get_params())

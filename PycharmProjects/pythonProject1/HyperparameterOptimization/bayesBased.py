from smac.facade.func_facade import fmin_smac
from bayes_src import branin
from time import time

start = time()
x, cost, _ = fmin_smac(func=branin, x0=[3.2, 4.5], bounds=[(-5, 10), (0, 15)], maxfun=500, rng=3)
end = time()

print("Time: {}".format(end-start))
print(x, cost)

import numpy as np
from xgboost import XGBClassifier
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter, \
    UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_ac_facade import SMAC4AC
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 自动调参算法
# Grid search（网格搜索）、Random search（随机搜索），
# Genetic algorithm（遗传算法）、Paticle Swarm Optimization（粒子群优化）、Bayesian Optimization（贝叶斯优化）、TPE、SMAC等。
# Grid search、Random search和Bayesian Optimization
cs = ConfigurationSpace()
learning_rate = UniformFloatH
yperparameter("learning_rate", 0.001, 0.1, default_value=0.1)
n_estimators = UniformIntegerHyperparameter("n_estimators", 100, 200, default_value=100)

cs.add_hyperparameters([learning_rate, n_estimators])
wbc_dataset = datasets.load_breast_cancer()


def xgboost_from_cfg(cfg):
    cfg = {k: cfg[k] for k in cfg if cfg[k]}

    clf = XGBClassifier(**cfg, eval_metric='auc', early_stopping_rounds=50, random_state=42)
    scores = cross_val_score(clf, wbc_dataset.data, wbc_dataset.target, cv=5)
    return 1 - np.mean(scores)


scenario = Scenario({"run_obj": "quality", "runcount-limit": 200, "cs": cs, "deterministic": "true"})

print("Please wait until optimization is finished")
smac = SMAC4AC(scenario=scenario, rng=np.random.RandomState(42), tae_runner=xgboost_from_cfg)
incumbent = smac.optimize()

print(incumbent)
inc_value = xgboost_from_cfg(incumbent)

print("Optimized Value: %.2f" % inc_value)

param_1 = []
param_2 = []
costs = []
for k, v in smac.runhistory.config_ids.items():
    param_1.append(k.values['learning_rate'])
    param_2.append(k.values['n_estimators'])
    costs.append(smac.runhistory.get_cost[v])

print(len(param_1), len(param_2), len(costs))

sc = plt.scatter(param_1, param_2, costs)
plt.colorbar(sc)
plt.show()

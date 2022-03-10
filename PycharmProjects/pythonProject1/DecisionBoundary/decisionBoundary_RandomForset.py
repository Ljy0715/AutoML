from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from decisionBoundaries import draw_decision_boundary
import matplotlib.cm as cm
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm.cool)
plt.show()

draw_decision_boundary(RandomForestClassifier(), X, y)

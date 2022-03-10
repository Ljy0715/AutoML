import matplotlib.pyplot as plt
from decisionBoundaries import draw_decision_boundary
import matplotlib.cm as cm
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier


X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm.cool)
plt.show()

draw_decision_boundary(LogisticRegression(), X, y)

draw_decision_boundary(RandomForestClassifier(), X, y)

draw_decision_boundary(svm.SVC(), X, y)

draw_decision_boundary(DecisionTreeClassifier(), X, y)

draw_decision_boundary(GradientBoostingClassifier(), X, y)

draw_decision_boundary(KNeighborsClassifier(), X, y)

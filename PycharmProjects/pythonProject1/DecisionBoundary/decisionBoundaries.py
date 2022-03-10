import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


def draw_decision_boundary(clf, X, y, h=.01, figsize=(9, 9), boundary_cmap=cm.winter, points_cmap=cm.cool):
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    print('Score: %0.3f' % score)
    pred = clf.predict(X_test)

    figure = plt.figure(figsize=figsize)

    ax = plt.subplot(1, 1, 1)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=boundary_cmap)

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=points_cmap, edgecolors='k')
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=points_cmap, alpha=0.6, edgecolors='k')
    figure.show()
    plt.show()




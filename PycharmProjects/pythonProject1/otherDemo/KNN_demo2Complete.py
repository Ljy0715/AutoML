import numpy as np
import operator
from scipy.spatial import distance


def nearest_neighbors_prediction(x, data, labels, k):
    distances = np.array([distance.euclidean(x, i) for i in data])
    # print('distance', distances)
    label_count = {}
    for i in range(k):
        label = labels[distances.argsort()[i]]
        print(label)
        label_count[label] = label_count.get(label, 0) + 1
    votes = sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)
    print(votes)
    return votes[0][0]


np.random.seed(23)

data = np.random.rand(100).reshape(20, 5)
print('data', data)
labels = np.random.choice(2, 20)
print('labels', labels)
x = np.random.rand(5)
print('x', x)
pred = nearest_neighbors_prediction(x, data, labels, k=2)

print(pred)

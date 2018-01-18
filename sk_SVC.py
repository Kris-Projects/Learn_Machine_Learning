from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

def loadData(filename):
    input = []  # store the feature vector x
    label = []  # store the label y
    with open(filename, 'r') as f:
        for line in f.readlines():
            tempArr = line.strip().split()
            input.append([float(tempArr[0]), float(tempArr[1])])
            label.append(float(tempArr[2]))
    return input, label

input, label = loadData('testSet.txt')
pos_1 = []
pos_2 = []
neg_1 = []
neg_2 = []
for i in range(len(label)):
    if label[i] == 1:
        pos_1.append(input[i][0])
        pos_2.append(input[i][1])
    else:
        neg_1.append(input[i][0])
        neg_2.append(input[i][1])

X = np.array(input)
y = np.array(label)
clf = SVC(C=0.6, kernel='linear', tol=0.001)
clf.fit(X, y)
support_1 = clf.support_vectors_[:, 0]
support_2 = clf.support_vectors_[:, 1]
w = clf.coef_[0]
b = clf.intercept_[0]
x_2 = np.linspace(-7, 5, 120)
x_1 = -b / w[0] - w[1] / w[0] * x_2

plt.scatter(pos_1, pos_2, label='$y = 1$')
plt.scatter(neg_1, neg_2, label='$y = -1$')
plt.scatter(support_1, support_2, c='red', label='support vectors')
plt.plot(x_1, x_2, c='green', label='separating hyperplane')
plt.legend()
plt.show()
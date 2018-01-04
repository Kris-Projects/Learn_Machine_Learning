import numpy as np


# define the function to load training set from file
def loadData(filename):
    input = []  # store the feature vector x
    label = []  # store the label y
    with open(filename, 'r') as f:
        for line in f.readlines():
            tempArr = line.strip().split()
            input.append([float(tempArr[0]), float(tempArr[1])])
            label.append(float(tempArr[2]))
    return input, label


# define the function to select alpha_j randomly after choosing alpha_i
def randSelectJ(i, m):  # m is the size of the training examples
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j


# define the function to make sure the updated alpha lying in the bounds
def clipAlpha(alpha_j, H, L):  # L is lower bound, H is upper bound
    if alpha_j > H:
        alpha_j = H
    elif alpha_j < L:
        alpha_j = L
    return alpha_j



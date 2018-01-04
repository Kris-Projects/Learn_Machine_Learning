import numpy as np


# define the function to load training set from file
def loadData(filename):
    input = []  # store the feature vector x
    label = []  # store the label y
    with open(filename, 'r') as f:
        for line in f.readlines():
            tempArr = line.strip().split()
            input.append([float(tempArr[0]), float(tempArr[1])])
            label.append([float(tempArr[2])])
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


# the simplified SMO algorithm
def SMO_Simplified(input, label, C, tol, iter_max):
    # change the list type to numpy array for matrix calculation
    x_data = np.array(input)
    y_data = np.array(label)

    # m is the size of training data, n is the number of input attributes
    m, n = x_data.shape

    # initialize the intercept term, the iteration time and all alphas to 0
    b = 0
    iter = 0
    alpha = np.zeros(shape=(m, 1))

    # before the time of empty iterations reaches iter_max, do the following iteration
    while iter < iter_max:
        alpha_pair_changed = 0
        for i in range(m): # run through the whole set to find alphas not satisfying KKT conditions
            # calculate E_i for each example based on (5.6.7)
            E_i = np.transpose(alpha * y_data).dot(x_data.dot(np.transpose(x_data[i]))) + b - y_data[i]
            # check if the i-th example satisfies KKT conditions
            if (label[i] * E_i < -tol and alpha[i] < C) or (label[i] * E_i > tol and alpha[i] > 0):
                # randomly select an alpha_j from the rest of the examples
                j = randSelectJ(i, m)
                # calculate E_j
                E_j = np.transpose(alpha * y_data).dot(x_data.dot(np.transpose(x_data[j]))) + b - y_data[j]

                # store the old alpha_i and alpha_j
                alpha_i_old = alpha[i].copy()
                alpha_j_old = alpha[j].copy()

                # set the upper and lower bounds
                if (y_data[i] != y_data[j]):
                    L = max((0, alpha[j] - alpha[i]))
                    H = min((C, C + alpha[j] - alpha[i]))
                else:
                    L = max((0, alpha[j] + alpha[i] - C))
                    H = min(C, alpha[j] + alpha[i])

                # if L = H, simply continue to the next iteration
                if L == H:
                    continue

                # calculate eta based on (5.6.8)
                eta = 2 * (np.transpose(x_data[i]).dot(x_data[j])) - np.transpose(x_data[i]).dot(x_data[i]) - np.transpose(x_data[j]).dot(x_data[j])

                # if eta >= 0, simply continue to the next iteration
                if eta >= 0:
                    continue

                # calculate the unclipped alpha_j_new
                alpha[j] -= y_data[j] * (E_i - E_j) / eta

                # clip alpha_j
                alpha[j] = clipAlpha(alpha[j], H, L)

                # check if alpha_j is accurate enough
                if abs(alpha[j] - alpha_j_old) < 0.00001:
                    continue

                # calculate updated alpha_i based on (5.6.10)
                alpha[i] += y_data[i] * y_data[j] * (alpha_j_old - alpha[j])

                # calculate the two threshold b_1 and b_2 based on (5.6.13) and (5.6.14)
                b_1 = b - E_i - y_data[i] * (alpha[i] - alpha_i_old) * (np.transpose(x_data[i]).dot(x_data[i])) -\
                    y_data[j] * (alpha[j] - alpha_j_old) * (np.transpose(x_data[j]).dot(x_data[j]))
                b_2 = b - E_j - y_data[i] * (alpha[i] - alpha_i_old) * (np.transpose(x_data[i]).dot(x_data[i])) -\
                    y_data[j] * (alpha[j] - alpha_j_old) * (np.transpose(x_data[j]).dot(x_data[j]))

                # decide which threshold to use
                if alpha[i] > 0 and alpha[i] < C:
                    b = b_1
                elif alpha[j] > 0 and alpha[j] < C:
                    b = b_2
                else:
                    b = (b_1 + b_2) / 2

                alpha_pair_changed += 1

        if alpha_pair_changed == 0:
            iter += 1
        else:
            iter = 0

    return b, alpha

if __name__ == '__main__':
    input, label = loadData('testSet.txt')
    b, alpha = SMO_Simplified(input, label, 0.6, 0.001, 40)
    print("b = ", b)
    print("alpha:\n", alpha)
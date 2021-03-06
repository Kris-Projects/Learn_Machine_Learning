import numpy as np
import matplotlib.pyplot as plt

# help function to read data in
def loadData(filename):
    input = []  # store the feature vector x
    label = []  # store the label y
    with open(filename, 'r') as f:
        for line in f.readlines():
            tempArr = line.strip().split()
            input.append([float(tempArr[0]), float(tempArr[1])])
            label.append([float(tempArr[2])])
    return input, label

# construct the class of SVM
class SVM:
    # initialize the variables needed
    def __init__(self, input, label, C, tol, k_tup=('lin', 0)):
        self.x_data = np.array(input)
        self.y_data = np.array(label)
        self.C = C
        self.tol = tol
        self.m = (self.x_data.shape)[0]
        self.alpha = np.zeros((self.m, 1))
        self.b = 0
        # the first column is used as an indicator, '1' means the value will be changed
        self.E_cache = np.zeros((self.m, 2))
        # k_tup is used to specify the kind of kernel function
        self.k_tup = k_tup
        # K is positive semi-definite matrix to store all the values of kernel functions
        self.K = np.zeros((self.m, self.m))
        # based on the calculating process of K, we know that this matrix will not change
        # so we can fill in this matrix in the initial function
        self.fill_K()

    # fill in the matrix K based on the specified kernel function
    def fill_K(self):
        if self.k_tup[0] == 'lin':
            self.K = self.x_data.dot(self.x_data.T)
        elif self.k_tup[0] == 'rbf':
            for i in range(self.m):
                for j in range(self.m):
                    self.K[i][j] = np.exp((self.x_data[i] - self.x_data[j]).T.dot(
                        self.x_data[i] - self.x_data[j]) / (-2 * self.k_tup[1] ** 2))
        else:
            raise NameError('Kernel not recognized!')

    # every time we update alpha and b, the value of every E_i will be changed
    # so we need a certain function to calculate the present value of E_i we need
    def calc_Ek(self, k):
        return (self.alpha * self.y_data).T.dot(self.K[:, k]) + self.b - self.y_data[k]

    # the random selection of j will only be called if all the indicators in E_cache
    # is 0, which means we can't use a heuristic way to choose j
    def randSelectJ(self, i):
        j = i
        while (j == i):
            j = int(np.random.uniform(0, self.m))
        return j

    # the heuristic way of choosing j
    def select_J(self, i, E_i):
        max_k = -1
        max_delta_E = 0
        E_j = 0
        self.E_cache[i] = [1, E_i]

        # find the E values with non-zero indicators
        valid_E_list = np.nonzero(self.E_cache[:, 0])[0]
        if len(valid_E_list) > 1:
            for k in valid_E_list:
                if k == i:
                    continue
                E_k = self.calc_Ek(k)
                delta_E = abs(E_i - E_k)
                if delta_E > max_delta_E:
                    max_k = k
                    max_delta_E = delta_E
                    E_j = E_k
            return max_k, E_j
        else: # no valid E_j, so we choose j randomly from the rest of the data
            j = self.randSelectJ(i)
            E_j = self.calc_Ek(j)
        return j, E_j

    # since E_k will be changed every time, we need a function to update the cache
    def update_Ek(self, k):
        E_k = self.calc_Ek(k)
        self.E_cache[k] = [1, E_k]

    def clipAlpha(self, alpha_j, H, L):  # L is lower bound, H is upper bound
        if alpha_j > H:
            alpha_j = H
        elif alpha_j < L:
            alpha_j = L
        return alpha_j

    # this is the update process of the chosen pair of alpha's
    # the return value indicates whether this is a successful update
    # notice that the first "if" statement can also be placed in the "fit" function below
    def inner_loop(self, i):
        E_i = self.calc_Ek(i)

        # check if alpha_i violates the KKT conditions. If this kind of alpha exists,
        # we continue to choose alpha_j, otherwise we treat this as an unsuccessful loop
        if (self.y_data[i] * E_i < -self.tol and self.alpha[i] < self.C) or\
                (self.y_data[i] * E_i > self.tol and self.alpha[i] > 0):
            j, E_j = self.select_J(i, E_i)
            alpha_i_old = self.alpha[i].copy()
            alpha_j_old = self.alpha[j].copy()
            if self.y_data[i] != self.y_data[j]:
                L = max((0, self.alpha[j] - self.alpha[i]))
                H = min((self.C, self.C + self.alpha[j] - self.alpha[i]))
            else:
                L = max((0, self.alpha[i] + self.alpha[j] - self.C))
                H = min((self.C, self.alpha[i] + self.alpha[j]))

            if L == H:
                return 0

            eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]

            if eta >= 0:
                return 0

            self.alpha[j] -= self.y_data[j] * (E_i - E_j) / eta
            self.alpha[j] = self.clipAlpha(self.alpha[j], H, L)
            self.update_Ek(j)

            if abs(self.alpha[j] - alpha_j_old) < 0.00001:
                return 0

            self.alpha[i] += self.y_data[i] * self.y_data[j] * \
                             (alpha_j_old - self.alpha[j])
            self.update_Ek(i)
            b_1 = self.b - E_i - self.y_data[i] * (self.alpha[i] - alpha_i_old) * self.K[
                i, i] - \
                  self.y_data[j] * (self.alpha[j] - alpha_j_old) * self.K[j, i]
            b_2 = self.b - E_j - self.y_data[i] * (self.alpha[i] - alpha_i_old) * self.K[
                i, j] - \
                  self.y_data[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]

            if self.alpha[i] > 0 and self.alpha[i] < self.C:
                self.b = b_1
            elif self.alpha[j] > 0 and self.alpha[j] < self.C:
                self.b = b_2
            else:
                self.b = (b_1 + b_2) / 2

            return 1

        else:
            return 0

    def fit(self, max_iter):
        iter = 0

        # for the first iteration, we can search the whole training set for alpha_i
        entire_set = True
        alpha_pair_changed = 0

        # this loop's end condition is that the number of unsuccessful loops
        # has reached max_iter
        while iter < max_iter and (alpha_pair_changed > 0 or entire_set):
            alpha_pair_changed = 0
            if entire_set:
                for i in range(self.m):
                    alpha_pair_changed += self.inner_loop(i)
                iter += 1
            else:
                # since it is not necessary to traverse the whole training set every time
                # we tend to check only the alphas between 0 and C
                non_bound_value  = np.nonzero((self.alpha > 0) * (self.alpha < self.C))[0]
                for i in non_bound_value:
                    alpha_pair_changed += self.inner_loop(i)
                iter += 1
            if entire_set:
                entire_set = False
            elif alpha_pair_changed == 0:
                entire_set = True

        return self.b, self.alpha

if __name__ == '__main__':
    input, label = loadData('testSet.txt')
    testSVM = SVM(input, label, 0.6, 0.001)
    b, alpha = testSVM.fit(40)
    print("b = ", b)
    print("alpha = ", alpha[alpha>0])
    # find support vectors
    pos_1 = []
    pos_2 = []
    neg_1 = []
    neg_2 = []
    support_1 = []
    support_2 = []
    for i in range(len(label)):
        if label[i][0] == 1:
            pos_1.append(input[i][0])
            pos_2.append(input[i][1])
        else:
            neg_1.append(input[i][0])
            neg_2.append(input[i][1])

        if alpha[i][0] != 0:
            support_1.append(input[i][0])
            support_2.append(input[i][1])

    # find separating hyper-plane
    x_2 = np.linspace(-7, 5, 120)
    x_1 = -((alpha * label).T.dot(input)[0][1] * x_2 + b) /\
            (alpha * label).T.dot(input)[0][0]

    # draw the scatter and hyperplane on figure
    plt.scatter(pos_1, pos_2, label='$y = 1$')
    plt.scatter(neg_1, neg_2, label='$y = -1$')
    plt.scatter(support_1, support_2, c='red', label='support vectors')
    plt.plot(x_1, x_2, c='green', label='separating hyperplane')
    plt.legend()

    plt.show()
import numpy as np
from sklearn import preprocessing
from numpy import random
import scipy.stats


# np.random.seed(0)
# random.seed(0)

class scaler:
    def __init__(self):
        self._mean = 0
        self._std = 0

    def fit_transform(self, traindata):
        self._mean = traindata.mean(axis=0)
        self._std = traindata.std(axis=0)
        return (traindata - self._mean) / (self._std + 1e-2)

    def transform(self, testdata):
        return (testdata - self._mean) / (self._std + 1e-2)


class node_generator:
    def __init__(self, whiten=False):
        self.Wlist = []
        self.blist = []
        self.nonlinear = 0
        self.whiten = whiten

    def sigmoid(self, data):
        return 1.0 / (1 + np.exp(-data))

    def linear(self, data):
        return data

    def tanh(self, data):
        return (np.exp(data) - np.exp(-data)) / (np.exp(data) + np.exp(-data))

    def relu(self, data):
        return np.maximum(data, 0)

    def orth(self, W):

        for i in range(0, W.shape[1]):
            w = np.mat(W[:, i].copy()).T
            w_sum = 0
            for j in range(i):
                wj = np.mat(W[:, j].copy()).T

                w_sum += (w.T.dot(wj))[0, 0] * wj  # [0,0]就是求矩阵相乘的一元数
            w -= w_sum
            w = w / (np.sqrt(w.T.dot(w)) + 1e-3)
            W[:, i] = np.ravel(w)

        return W

    def generator(self, shape, times):
        # np.random.seed(0)
        # random.seed(0)
        for i in range(times):
            W = 2 * random.random(size=shape) - 1
            if self.whiten == True:
                W = self.orth(W)
            b = 2 * random.random() - 1
            yield (W, b)

    def generator_nodes(self, data, times, batchsize, nonlinear):  # 将特征结点和增强结点构建起来
        self.Wlist = [elem[0] for elem in self.generator((data.shape[1], batchsize), times)]
        self.blist = [elem[1] for elem in self.generator((data.shape[1], batchsize), times)]

        self.nonlinear = {'linear': self.linear,
                          'sigmoid': self.sigmoid,
                          'tanh': self.tanh,
                          'relu': self.relu
                          }[nonlinear]
        nodes = self.nonlinear(data.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            nodes = np.column_stack((nodes, self.nonlinear(data.dot(self.Wlist[i]) + self.blist[i])))
        return nodes

    def transform(self, testdata):
        testnodes = self.nonlinear(testdata.dot(self.Wlist[0]) + self.blist[0])
        for i in range(1, len(self.Wlist)):
            testnodes = np.column_stack((testnodes, self.nonlinear(testdata.dot(self.Wlist[i]) + self.blist[i])))
        return testnodes

    def update(self, otherW, otherb):
        self.Wlist += otherW
        self.blist += otherb


class Lite_RVFL:
    def __init__(self,
                 Ne=10,
                 N2=10,
                 enhence_function='sigmoid',
                 reg=1,
                 theta = 1.002):
        self._Ne = Ne
        self._enhence_function = enhence_function
        self._reg = reg
        self._N2 = N2

        self.W = 0
        self.P = 0
        self.K = 0
        self.T = np.array([[]])
        self.theta = theta

        self.t = 0
        self.normalscaler = scaler()
        self.onehotencoder = preprocessing.OneHotEncoder(sparse=False)
        self.enhence_generator = node_generator(whiten=True)

    def fit(self, oridata, orilabel):
        N = len(orilabel)
        self.t = N

        data = self.normalscaler.fit_transform(oridata)
        label = self.onehotencoder.fit_transform(np.mat(orilabel).T)
        enhencedata = self.enhence_generator.generator_nodes(data, self._Ne, self._N2, self._enhence_function)
        inputdata = np.column_stack((data, enhencedata))

        r, w = inputdata.T.dot(inputdata).shape

        self.T = np.zeros((N, N))
        for i in range(N):
            self.T[i, i] = self.theta ** i
        self.P = np.linalg.inv(inputdata.T.dot(self.T.T).dot(self.T).dot(inputdata) + self._reg * np.eye(r))
        self.Q = inputdata.T.dot(self.T.T).dot(self.T).dot(label)
        self.W = self.P.dot(self.Q)


    def softmax_norm(self, array):
        exp_array = np.exp(array)
        exp_array[np.isinf(exp_array)] = 1
        sum_exp_array = np.sum(exp_array, axis=1, keepdims=True)
        softmax_array = exp_array / (sum_exp_array + 1e-6)
        return softmax_array

    def decode(self, Y_onehot):
        Y = []
        for i in range(Y_onehot.shape[0]):
            lis = np.ravel(Y_onehot[i, :]).tolist()
            Y.append(lis.index(max(lis)))
        return np.array(Y)

    def predict(self, testdata):
        logit = self.predict_proba(testdata)
        return self.decode(self.softmax_norm(logit))

    def predict_proba(self, testdata):
        testdata = self.normalscaler.transform(testdata)
        test_inputdata = self.transform(testdata)
        org_prediction = test_inputdata.dot(self.W)
        # print('self.W={}'.format(self.W))
        # print('org_prediction={}'.format(org_prediction))
        return self.softmax_norm(org_prediction)

    def transform(self, data):
        enhencedata = self.enhence_generator.transform(data)
        inputdata = np.column_stack((data, enhencedata))
        return inputdata

    def partial_fit(self, extratraindata, extratrainlabel):


        xdata = self.normalscaler.transform(extratraindata)
        xdata = self.transform(xdata)
        xlabel = self.onehotencoder.transform(np.mat(extratrainlabel).T)
        temp = self.theta**(2*self.t)*(xdata.dot(self.P)).dot(xdata.T)
        r, w = temp.shape

        self.P = self.P - self.theta**(2*self.t)*(((self.P.dot(xdata.T)).dot(np.linalg.inv(np.eye(r) + temp))).dot(xdata)).dot(self.P)
        self.Q = self.Q + self.theta**(2*self.t)*xdata.T.dot(xlabel)
        self.W = self.P.dot(self.Q)


        self.t = self.t + 1

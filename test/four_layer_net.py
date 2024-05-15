# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class BatchNormalization:
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        self.gamma = np.ones(input_size)
        self.beta = np.zeros(input_size)
        self.momentum = momentum
        self.eps = eps

        self.running_mean = np.zeros(input_size)
        self.running_var = np.zeros(input_size)

    def forward(self, x, train_flg=True):
        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + self.eps)
            xn = xc / std

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self.xc = xc
            self.std = std
            self.xn = xn
        else:
            xc = x - self.running_mean
            xn = xc / (np.sqrt(self.running_var + self.eps))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        dxn = dout * self.gamma
        dgamma = np.sum(dout * self.xn, axis=0)
        dbeta = np.sum(dout, axis=0)

        dx = (1. / self.std) * (dxn - np.mean(dxn, axis=0) - self.xn * np.mean(dxn * self.xn, axis=0))

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

class FourLayerNet:
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, hidden_size4, output_size):
        # He 초기화에 따라 가중치 초기화
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size1) * np.sqrt(2 / input_size)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['W2'] = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2 / hidden_size1)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['W3'] = np.random.randn(hidden_size2, hidden_size3) * np.sqrt(2 / hidden_size2)
        self.params['b3'] = np.zeros(hidden_size3)
        self.params['W4'] = np.random.randn(hidden_size3, hidden_size4) * np.sqrt(2 / hidden_size3)
        self.params['b4'] = np.zeros(hidden_size4)
        self.params['W5'] = np.random.randn(hidden_size4, output_size) * np.sqrt(2 / hidden_size4)
        self.params['b5'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['BatchNorm1'] = BatchNormalization(hidden_size1)
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['BatchNorm2'] = BatchNormalization(hidden_size2)
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['BatchNorm3'] = BatchNormalization(hidden_size3)
        self.layers['Relu3'] = Relu()
        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['BatchNorm4'] = BatchNormalization(hidden_size4)
        self.layers['Relu4'] = Relu()
        self.layers['Affine5'] = Affine(self.params['W5'], self.params['b5'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x, train_flg=True):
        for layer in self.layers.values():
            if isinstance(layer, BatchNormalization):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    # x: 입력 데이터, t: 정답 레이블
    def loss(self, x, t, train_flg=True):
        y = self.predict(x, train_flg)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x, train_flg=False)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t, train_flg=True)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grads['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grads['b3'] = numerical_gradient(loss_W, self.params['b3'])
        grads['W4'] = numerical_gradient(loss_W, self.params['W4'])
        grads['b4'] = numerical_gradient(loss_W, self.params['b4'])
        grads['W5'] = numerical_gradient(loss_W, self.params['W5'])
        grads['b5'] = numerical_gradient(loss_W, self.params['b5'])

        return grads

    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
        grads['W4'], grads['b4'] = self.layers['Affine4'].dW, self.layers['Affine4'].db
        grads['W5'], grads['b5'] = self.layers['Affine5'].dW, self.layers['Affine5'].db

        # BatchNorm의 파라미터를 저장
        grads['gamma1'] = self.layers['BatchNorm1'].dgamma
        grads['beta1'] = self.layers['BatchNorm1'].dbeta
        grads['gamma2'] = self.layers['BatchNorm2'].dgamma
        grads['beta2'] = self.layers['BatchNorm2'].dbeta
        grads['gamma3'] = self.layers['BatchNorm3'].dgamma
        grads['beta3'] = self.layers['BatchNorm3'].dbeta
        grads['gamma4'] = self.layers['BatchNorm4'].dgamma
        grads['beta4'] = self.layers['BatchNorm4'].dbeta

        return grads

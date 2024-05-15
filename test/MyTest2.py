# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from common.util import smooth_curve
from four_layer_net import FourLayerNet
from optimizer import Momentum

# MNIST 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

train_size = x_train.shape[0]
batch_size = 1000
max_iterations = 10000

# 실험용 설정
optimizer = Momentum()

network = FourLayerNet(input_size=784, hidden_size1=15, hidden_size2=15,
                       hidden_size3=15,hidden_size4=15, output_size=10)
train_loss = []

# 훈련 시작
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    loss = network.loss(x_batch, t_batch)
    train_loss.append(loss)

    if i % 500 == 0 or i == max_iterations-1:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        print("iteration:", i)
        print("AdaGrad loss:", loss)
        print("train acc:", train_acc, "test acc:", test_acc)

# 그래프 그리기
x = np.arange(max_iterations)
plt.plot(x, smooth_curve(train_loss), label='optimizer', marker='s', markevery=100)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()

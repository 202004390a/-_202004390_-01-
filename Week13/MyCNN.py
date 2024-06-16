# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import matplotlib.pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from simple_convnet import SimpleConvNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, normalize=True, one_hot_label=True)

# 원-핫 인코딩된 레이블을 정수형 레이블로 변환
t_train = np.argmax(t_train, axis=1)
t_test = np.argmax(t_test, axis=1)

# SimpleConvNet 초기화
network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                        hidden_size1=100, hidden_size2=50, output_size=10, weight_init_std=0.009)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.015

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    grad = network.gradient(x_batch, t_batch)  # 오차역전파법 방식(훨씬 빠르다)

    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f"train accuracy: {train_acc}, test accuracy: {test_acc}")

epochs = len(train_acc_list)
x = np.arange(epochs)
plt.plot(x, train_acc_list, label='train acc', marker='o')
plt.plot(x, test_acc_list, label='test acc', marker='s')
plt.xlabel("epochs")
plt.ylabel("accuracy")

plt.ylim(0, 1)
plt.legend()
plt.show()

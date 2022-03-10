from sklearn import datasets
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.python.client import device_lib

print("版本:", tf.__version__)
print("型号:", device_lib.list_local_devices())

x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

print(type(x_data))
print(type(y_data))

print(x_data)
print(y_data)

# x_data = DataFrame(x_data, columns=['花萼长度', '花萼宽度', '花瓣长度', '花瓣宽度'])
# pd.set_option('display.unicode.east_asian_width', True)  # 设置列名对齐


# x_data['类别'] = y_data  # 在x——data后加入一列y
# print(x_data)
np.random.seed(116)
np.random.shuffle(x_data)
# 将x打乱
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

# 分为测试集和训练集

x_train = x_data[:-30]
y_train = y_data[:-30]

x_test = x_data[-30:]  # 从第30个数据取到最后一个
y_test = y_data[-30:]

# 将测试集变成32位浮点防止报错
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)


# 配成输入特征和标签每次训练32个
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(16)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(16)

w1 = tf.Variable(tf.random.truncated_normal([4, 11], stddev=0.1, seed=1))
b1 = tf.Variable(tf.random.truncated_normal([1, 11], stddev=0.1, seed=1))

w2 = tf.Variable(tf.random.truncated_normal([11, 3], stddev=0.1, seed=2))
b2 = tf.Variable(tf.random.truncated_normal([1, 3], stddev=0.1, seed=2))

epoch = 150
lr = 0.2
test_acc = []
loss_all = 0
train_loss_result = []

for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):
        with tf.GradientTape() as tape:
            # y = tf.matmul(x_train, w1) + b1
            # y = tf.nn.softmax(y)
            # y_ = tf.one_hot(y_train, depth=3)
            # print(x_train.shape)

            y1 = tf.matmul(x_train, w1) + b1
            y = tf.matmul(y1, w2) + b2
            y = tf.nn.softmax(y)
            y_ = tf.one_hot(y_train, depth=3)
            # print(y_)

            loss = tf.reduce_mean(tf.square(y_-y))
            loss_all += loss.numpy()

        grads = tape.gradient(loss, [w1, b1, w2, b2])
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
    print("Epoch{}, loss:{}".format(epoch, loss_all/4))
    train_loss_result.append(loss_all / 4)
    loss_all = 0

    total_correct, total_number = 0, 0

    for x_test, y_test in test_db:
        # 测试，将y归一化
        y1 = tf.matmul(x_test, w1)+b1
        y = tf.matmul(y1, w2) + b2
        y = tf.nn.softmax(y)
        # 寻找最大的y值
        pred = tf.argmax(y, axis=1)  # y为行向量
        # 转为和y_test一样的数据
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 将正确结果存在correct中
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        # correct相加
        correct = tf.reduce_sum(correct)
        total_correct += int(correct)
        total_number += x_test.shape[0]
        print(total_correct)
    acc = total_correct / total_number
    test_acc.append(acc)
    print('Test_acc:', acc)
    print('-------------------')

plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_result, label='$Loss$')
plt.legend()
plt.show()

plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label='$Accuracy$')
plt.legend()
plt.show()


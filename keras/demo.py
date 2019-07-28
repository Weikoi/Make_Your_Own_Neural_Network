import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.models import Sequential
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train / 255.
x_test = x_test / 255.

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

model = Sequential()
model.add(Conv2D(20, (5, 5), strides=(1, 1), input_shape=(28, 28, 1), padding='valid', activation='relu',
                 kernel_initializer='uniform'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), strides=(1, 1), padding='valid', activation='relu', kernel_initializer='uniform'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile('sgd', loss='categorical_crossentropy', metrics=['accuracy'])  # 随机梯度下降

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.25)  # 获取数据

#########画图
acc = history.history['acc']  # 获取训练集准确性数据
val_acc = history.history['val_acc']  # 获取验证集准确性数据
loss = history.history['loss']  # 获取训练集错误值数据
val_loss = history.history['val_loss']  # 获取验证集错误值数据
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Trainning acc')  # 以epochs为横坐标，以训练集准确性为纵坐标
plt.plot(epochs, val_acc, 'b', label='Vaildation acc')  # 以epochs为横坐标，以验证集准确性为纵坐标
plt.legend()  # 绘制图例，即标明图中的线段代表何种含义

plt.figure()  # 创建一个新的图表
plt.plot(epochs, loss, 'bo', label='Trainning loss')
plt.plot(epochs, val_loss, 'b', label='Vaildation loss')
plt.legend()
plt.show()
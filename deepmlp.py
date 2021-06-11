import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# csv 형식으로 된 파일 가져오기
mnist_train = "/content/train.csv"
mnist_test = "/content/test.csv"

# csv 형식으로 된 파일 읽어오기
raw_train_data = pd.read_csv(mnist_train)
raw_test_data = pd.read_csv(mnist_test)

x_train = np.array(raw_train_data.drop("label", axis=1)).astype(float)
y_train = np.array(raw_train_data["label"])

x_test = np.array(raw_test_data.drop("id", axis=1)).astype(float)
y_test = np.array(raw_test_data["id"])

n_input = 784
n_hidden1 = 1024
n_hidden2 = 512
n_hidden3 = 512
n_hidden4 = 512
n_output = 10

mlp = Sequential()
mlp.add(Dense(units=n_hidden1, activation='tanh',input_shape=(n_input,), kernel_initializer='random_uniform', bias_initializer='zero'))
mlp.add(Dense(units=n_hidden2, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros'))
mlp.add(Dense(units=n_hidden3, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros'))
mlp.add(Dense(units=n_hidden4, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros'))
mlp.add(Dense(units=n_output, activation='tanh', kernel_initializer='random_uniform', bias_initializer='zeros'))

mlp.compile(loss='mse', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
hist=mlp.fit(x_train, y_train, batch_size=128, epochs=30, validation_data=(x_test, y_test), verbose=2)

res = mlp.evaluate(x_test, y_test, verbose=0)
print("Accuracy is ", res[1]*100, "%")

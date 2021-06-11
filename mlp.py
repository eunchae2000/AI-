import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

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

# random_state=42 for reproducibility
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.6)

mlp = MLPClassifier(hidden_layer_sizes=(100),
                    learning_rate_init=0.001,
                    batch_size=32,
                    solver='sgd',
                    verbose=True)

mlp.fit(x_train, y_train)

res = mlp.predict(x_test)

conf = np.zeros((10, 10))
for i in range(len(res)):
    conf[res[i]][y_test[i]] += 1
print(conf)

correct = 0
for i in range(10):
    correct += conf[i][i]
accuracy = correct/len(res)
print("Accuracy is ", accuracy*100, "%")

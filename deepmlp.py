import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

TRAIN_DATA_PATH = "C:\Users\coco1\Desktop\AI시스템설계\train.csv"
TEST_DATA_PATH = "C:\Users\coco1\Desktop\AI시스템설계\test.csv"

X_train = np.array(raw_train_data.drop("label", axis=1)).astype(float)
y_train = np.array(raw_train_data["label"])

X_test = np.array(raw_test_data.drop("id", axis=1)).astype(float)
y_test = np.array(raw_test_data["id"])

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)


print(f"X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")
print(f"X_val Shape: {X_val.shape}, y_val Shape: {y_val.shape}")
print(f"X_test Shape: {X_test.shape}, y_test Shape: {y_test.shape}")

np.random.seed(42)
tf.random.set_seed(42)

mlp_model = Sequential()
mlp_model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=[28, 28], name="input_layer"),
                tf.keras.layers.Dense(1024, activation="relu", name="hidden_layer1"),
                tf.keras.layers.Dense(512, activation="relu", name="hidden_layer2"),
                tf.keras.layers.Dense(512, activation="relu", name="hidden_layer3"),
                tf.keras.layers.Dense(512, activation="relu", name="hidden_layer4"),
                tf.keras.layers.Dense(10, activation="softmax", name="output_layer")
])

# compile MLP model
mlp_model.compile(loss="mse",
                  optimizer=Adam(learning_rate=0.001),
                  metrics=["accuracy"])

# display a breakdown of the MLP model
mlp_model.summary()

val_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# train the MLP
training_progress = mlp_model.fit(X_train, y_train, epochs=1000,
                                  batch_size=128,
                                  validation_data=(X_val, y_val),
                                  verbose = 2,
                                  callbacks=[val_stop])

res = mlp_model.evaluate(X_test, y_test, verbose = 0)
print("Accuracy is ", res[1]*100)
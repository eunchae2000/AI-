import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

TRAIN_DATA_PATH = "/train.csv"
TEST_DATA_PATH = "/test.csv"

raw_train_data = pd.read_csv(TRAIN_DATA_PATH)
raw_test_data = pd.read_csv(TEST_DATA_PATH)

plot_ten_df = raw_train_data.drop("label", axis=1).iloc[0:10, :]
plt.rcParams['figure.figsize'] = [15, 15]

# visualize the first 10 digits in the train set 
for index in range(10):
    plt.subplot(1, 10, index+1)
    # reshape pixel arragement to 28 x 28
    digit_array = np.asarray(plot_ten_df.iloc[index]).reshape(28, 28)
    plt.imshow(digit_array, cmap="binary")
    plt.title(raw_train_data["label"].iloc[index], fontsize=16)
    plt.axis("off")

X_train = np.array(raw_train_data.drop("label", axis=1)).astype(float)
y_train = np.array(raw_train_data["label"])

X_test = np.array(raw_test_data.drop("id", axis=1)).astype(float)
y_test = np.array(raw_test_data["id"])

# divide by 255 to normalize 
# reshape arrays to 28 x 28 to match the pixel format
# X_train = (X_train / 255).reshape(60000, 28, 28)
# X_test = (X_test / 255).reshape(10000, 28, 28) 

from sklearn.model_selection import train_test_split

# random_state=42 for reproducibility
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

# verify the set sizes are as expected
print(f"X_train Shape: {X_train.shape}, y_train Shape: {y_train.shape}")
print(f"X_val Shape: {X_val.shape}, y_val Shape: {y_val.shape}")
print(f"X_test Shape: {X_test.shape}, y_test Shape: {y_test.shape}")

np.random.seed(42)
tf.random.set_seed(42)

# build the MLP architecture
mlp_model = tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=[28, 28], name="input_layer"),
                tf.keras.layers.Dense(150, activation="relu", name="hidden_layer1"),
                tf.keras.layers.Dense(100, activation="relu", name="hidden_layer2"),
                tf.keras.layers.Dense(50, activation="relu", name="hidden_layer3"),
                tf.keras.layers.Dense(10, activation="softmax", name="output_layer")
])

# compile MLP model
mlp_model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="sgd",
                  metrics=["accuracy"])

# display a breakdown of the MLP model
mlp_model.summary()

val_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# train the MLP
training_progress = mlp_model.fit(X_train, y_train, epochs=1000,
                                  validation_data=(X_val, y_val),
                                  callbacks=[val_stop])
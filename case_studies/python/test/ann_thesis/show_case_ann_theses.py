import pandas as pd
import numpy as np
from timeit import default_timer as timer
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import rsqtoa

df = pd.read_csv('./../../train_sample_10000.csv', index_col=[0])

X = df.iloc[:, :2].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

y_train = np.expand_dims(y_train, axis=1)
y_val = np.expand_dims(y_val, axis=1)

start = timer()

model = rsqtoa.FunctionApproximator()
# model.add(tf.keras.Input(shape=(2,)))
model.add(tf.keras.layers.Dense(100, input_dim=2, activation='elu'))
model.add(tf.keras.layers.Dense(1000, activation='elu'))
model.add(tf.keras.layers.Dense(1000, activation='elu'))
model.add(tf.keras.layers.Dense(1000, activation='elu'))
model.add(tf.keras.layers.Dense(10, activation='elu'))
model.add(tf.keras.layers.Dense(1, activation='elu'))
model.summary()

print('Total compile time ::', timer() - start)


start = timer()
loss_hist = rsqtoa.train_model(model, X_train, y_train, X_val, y_val)
print('Total training time ::', timer() - start)


# print(loss_hist)

# model = Sequential()
# model.add(Dense(100, input_dim=2, activation='elu'))
# model.add(Dense(1000, activation='elu'))
# model.add(Dense(1000, activation='elu'))
# model.add(Dense(1000, activation='elu'))
# model.add(Dense(10, activation='elu'))
# model.add(Dense(1, activation='linear'))
#
# model.summary()
#
# model.compile(loss='mse', optimizer='adam', metrics=[tf.keras.metrics.RootMeanSquaredError()])
# print('Total compile time ::', timer() - start)
# start = timer()
# history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=40)
# print('Total training time ::', timer() - start)

y_pred = model.predict(X_test)

print('Mean Squared Error (Train out of sample):: ', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (Train out of sample):: ', mean_squared_error(y_test, y_pred, squared=False))

plt.plot(loss_hist['train_loss_points'][1])
plt.plot(loss_hist['val_test_loss_points'][1])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save('model')

# reconstructed_model = keras.models.load_model("model")

df_test = pd.read_csv('./../../test_sample_1000.csv', index_col=[0])

X_test = df.iloc[:, :2].values
y_test = df.iloc[:, -1].values

start = timer()
y_pred = model.predict(X_test)
print('Total prediction time ::', timer() - start)

print('Mean Squared Error (out of sample):: ', mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error (out of sample):: ', mean_squared_error(y_test, y_pred, squared=False))

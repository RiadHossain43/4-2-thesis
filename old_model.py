import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv(
    "https://raw.githubusercontent.com/RiadHossain43/data/master/pcf_mode_soln_data.csv")
# print(df)

# reverse (fiber parameter pred)
x = df.drop(['no-of-\nrings', 'diaBYpitch', 'pitch\n(um)', 'wl\n(um)',
            'conf-loss-in-log10\n(dB/cm)', 'core-ref-index-at-wl-1.55um', 'clad-ref-index'], axis=1)
x = x.values
# print(x)
y = df.drop(['Aeff\n(um^2)', 'conf-loss\n(dB/cm)', 'conf-loss-in-log10\n(dB/cm)',
            'dispersion\n(ps/km.nm)', 'clad-ref-index', 'core-ref-index-at-wl-1.55um', 'neff'], axis=1)
y = y.values
print(y)

# # forward (fiber charectaristic pred)
# x = df.drop(['Aeff\n(um^2)', 'conf-loss\n(dB/cm)', 'conf-loss-in-log10\n(dB/cm)',
#              'dispersion\n(ps/km.nm)', 'clad-ref-index', 'core-ref-index-at-wl-1.55um', 'neff'], axis=1)
# x = x.values
# y = df.drop(['no-of-\nrings', 'diaBYpitch', 'pitch\n(um)', 'wl\n(um)',
#             'conf-loss-in-log10\n(dB/cm)', 'core-ref-index-at-wl-1.55um', 'clad-ref-index'], axis=1)
# y = y.values

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=.2, random_state=100)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = keras.Sequential([
    # Input layer
    keras.layers.Input(shape=(x_train.shape[1],)),
    # Hidden layer with 64 neurons and ReLU activation
    keras.layers.Dense(64, activation='relu'),
    # Hidden layer with 32 neurons and ReLU activation
    keras.layers.Dense(64, activation='relu'),
    # Output layer with the same number of neurons as outputs
    keras.layers.Dense(y_train.shape[1])
])

# # Compile the model
# print("Comipling model...")
# model.compile(optimizer='adam', loss='mean_squared_error',
#               metrics=["accuracy"])
# model.fit(x_train, y_train, epochs=1000, batch_size=32,
#           validation_data=(x_test, y_test))
# pickle.dump(model, open('E:/thesis/works-and-data/ml/00-model', 'wb'))

# Make predictions on new data
loaded_model = pickle.load(open('E:/thesis/works-and-data/ml/00-model', 'rb'))
loss = loaded_model.evaluate(x_test, y_test)
print(f"Mean Squared Error on Test Data: {loss}")
y_pred = loaded_model.predict(x_test)
print("Predicted Output:", y_pred)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)
mse = mean_squared_error(y_test, y_pred)
print("Mean Sqaured Error:", mse)
r_squared = r2_score(y_test, y_pred)
print("R Sqaured Error:", r_squared)

# Plotting Scatter Plots for Each Output Variable
for i in range(y_test.shape[1]):
    plt.figure(figsize=(4, 3))
    plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
    plt.xlabel(f"True Value {i+1}")
    plt.ylabel(f"Predicted Value {i+1}")
    plt.title(f"Scatter Plot for Output Variable {i+1}")
    plt.grid()
    plt.show()









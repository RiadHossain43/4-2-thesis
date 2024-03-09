import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import numpy as np
df = pd.read_csv("./dataset.csv")
# print(df)

# reverse (fiber parameter pred)
x = df.drop(["hole_diameter(um)","pitch(um)","fiber_diameter(um)", "elips_semi_major_a" , "elips_semi_minor_b",  "wevelength(um)","pml(um)"], axis=1)
x = x.values
# y = df.drop(["real_effective_mode_index_x","real_effective_mode_index_y",'birefringence','conf_x','conf_y','pml(um)'], axis=1)
y = df.loc[:, ["hole_diameter(um)"]]
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=.3, random_state=100)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)


# print(y_train)
model = keras.Sequential([
    # Input layer
    keras.layers.Input(shape=(x_train.shape[1],)),
    # Hidden layer with 16 neurons and ReLU activation
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    # Hidden layer with 16 neurons and ReLU activation
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='relu'),
    # Output layer with the same number of neurons as outputs
    keras.layers.Dense(y_train.shape[1])
])

# Compile the model
print("Comipling model...")
model.compile(optimizer='adam', loss='mean_squared_error',
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=70, batch_size=16,
          validation_data=(x_test, y_test))
# pickle.dump(model, open('E:/thesis/works-and-data/ml/01-model', 'wb'))

# Make predictions on new data
# loaded_model = pickle.load(open('E:/thesis/works-and-data/ml/00-model', 'rb'))
loaded_model = model
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
# for i in range(y_test.shape[1]):
#     plt.figure(figsize=(4, 3))
#     plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.5)
#     plt.xlabel(f"True Value {i+1}")
#     plt.ylabel(f"Predicted Value {i+1}")
#     plt.title(f"Scatter Plot for Output Variable {i+1}")
#     plt.grid()
#     plt.show()
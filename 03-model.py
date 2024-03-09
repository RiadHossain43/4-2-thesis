import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, precision_score
import numpy as np
df = pd.read_csv("./dataset.csv")
# print(df)

# reverse (fiber parameter pred)
x_input = df.loc[:,["wevelength(um)","real_effective_mode_index_x","real_effective_mode_index_y","birefringence","conf_x","conf_y"]]
x = x_input.values
# y = df.drop(["real_effective_mode_index_x","real_effective_mode_index_y",'birefringence','conf_x','conf_y','pml(um)'], axis=1)
y_input = df.loc[:, ["hole_diameter(um)","pitch(um)","elips_semi_major_a","elips_semi_minor_b"]]
y = y_input
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
    keras.layers.Dense(16, activation='tanh'),
    keras.layers.Dropout(0.2),
    # Hidden layer with 16 neurons and ReLU activation
    keras.layers.Dense(16, activation='tanh'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation='tanh'),
    keras.layers.Dropout(0.2),
    # Output layer with the same number of neurons as outputs
    keras.layers.Dense(y_train.shape[1])
])


# Compile the model
print("Comipling model...")
model.compile(optimizer='adam', loss='mean_squared_error',
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=50, batch_size=8,
          validation_data=(x_test, y_test))
pickle.dump(model, open('E:/thesis/works-and-data/ml/03-model', 'wb'))

# Make predictions on new data
loaded_model = pickle.load(open('E:/thesis/works-and-data/ml/03-model', 'rb'))
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

# for i in range(y_test.shape[1]):
#     plt.figure(figsize=(4, 3))
#     plt.scatter(y_test.values[:, i], y_pred[:, i], color='blue')
#     plt.plot([y_test.values[:, i].min(), y_test.values[:, i].max()], [y_test.values[:, i].min(), y_test.values[:, i].max()], color='red', linestyle='--')
#     plt.xlabel('True Values')
#     plt.ylabel('Predicted Values')
#     plt.title(f'True vs Predicted Values for Target {i+1}')
#     plt.show()

# Create a scatter plot for each output variable
# for i in range(y_test.shape[1]):
#     plt.figure(figsize=(4, 4))
#     plt.scatter(y_test.iloc[:, i], y_pred[:, i], color='blue', alpha=0.5)
#     plt.xlabel(f'True Values {y_input.columns[i]}')
#     plt.ylabel(f'Predicted Values {y_input.columns[i]}')
#     plt.title(f'Scatter Plot for output {y_input.columns[i]}. Distance unit(um)')
#     plt.grid(True)
#     plt.show()


for i in range(y_test.shape[1]):
    plt.figure(figsize=(4, 4))
    plt.scatter(y_test.iloc[:, i], y_pred[:, i], color='blue', alpha=0.5)
    plt.xlabel(f'True Values {y_input.columns[i]}')
    plt.ylabel(f'Predicted Values {y_input.columns[i]}')
    plt.title(f'Scatter Plot for output {y_input.columns[i]}')
    
    # Remove grid
    plt.grid(False)
    
    # Customize axis
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)  # Adjust line width as needed
    ax.spines['bottom'].set_linewidth(0.5)  # Adjust line width as needed
    
    plt.show()
import pickle
import pandas as pd
df = pd.read_csv("./testdata.csv")
x_input = df.loc[:,["wevelength(um)","real_effective_mode_index_x","real_effective_mode_index_y","birefringence","conf_x","conf_y"]]
x_test = x_input.values
loaded_model = pickle.load(open('E:/thesis/works-and-data/ml/03-model', 'rb'))
y_pred = loaded_model.predict(x_test)
print("Predicted Output:\n", y_pred)
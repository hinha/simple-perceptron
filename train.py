from main import Perceptron, Perceptron2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

# baca data csv
# csv_data = pd.read_excel("transfusion_data.csv", delimiter=',', header=0)
csv_data = pd.read_excel("transfusion_data.xlsx").drop(
    columns=['No']).head(500)

df = csv_data
target = df['whether he/she donated blood in March 2007']
rerata_recen = df['Recency (months)'].mean()
rerata_time = df['Frequency (times)'].mean()
rerata_monet = df['Monetary (c.c. blood)'].mean()
rerata_month = df["Time (months)"].mean()
rerata_donate = df['whether he/she donated blood in March 2007'].mean()


kolom_count = df.shape[0]

col1, col2, col3, col4 = [], [], [], []
for i in range(0, len(df)):
    col1.append(math.sqrt(((df.iloc[i]['Recency (months)'] - rerata_recen) *
                           (df.iloc[i]['Recency (months)'] - rerata_recen)) / kolom_count))
    col2.append(math.sqrt(((df.iloc[i]['Frequency (times)'] - rerata_time) *
                           (df.iloc[i]['Frequency (times)'] - rerata_time)) / kolom_count))
    col3.append(math.sqrt(((df.iloc[i]['Monetary (c.c. blood)'] - rerata_monet) *
                           (df.iloc[i]['Monetary (c.c. blood)'] - rerata_monet)) / kolom_count))
    col4.append(math.sqrt(((df.iloc[i]["Time (months)"] - rerata_month) *
                           (df.iloc[i]["Time (months)"] - rerata_month)) / kolom_count))

new_datadf = pd.DataFrame({
    'col1': col1,
    'col2': col2,
    'col3': col3,
    'col4': col4,
    'target': target
})


data_target = np.array(target)
data_target = data_target.astype(int)
data = np.array(new_datadf)  # konversi data csv menjadi array
data = data.astype(int)
n_feature = len(data[0, :]) - 1
n_data = len(data[0, :])
print(n_data)


# membagi data: data latih dan uji
rasio_data_latih = 0.7
n_data_latih = int(n_data * rasio_data_latih)
data_latih = data[:n_data_latih, :]

data_uji = data[n_data_latih:, :]
n_data_uji = len(data_uji[:, 0])

print(n_data_uji)

datasets = new_datadf.values.tolist()
perceptron2 = Perceptron2(data_latih=n_data_latih,
                          epochs=220, learning_rate=0.8)
scores = perceptron2.evaluate_algorithm(datasets)
bobot = perceptron2.weights
print()
print('Scores: %s' % scores)


# print grafik MSE hasil training
plt.title("Mean Squared Error hasil Test")
plt.plot(perceptron2.MSE)
plt.autoscale(enable=True, axis='both', tight=None)
plt.show(block=False)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

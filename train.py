from main import Perceptron, Perceptron2
import pandas as pd
import numpy as np

# baca data csv
# csv_data = pd.read_excel("transfusion_data.csv", delimiter=',', header=0)
csv_data = pd.read_excel("transfusion_data.xlsx",
                         delimiter=",", header=0).drop(columns=['No'])

df = csv_data
target = df['whether he/she donated blood in March 2007']
rerata_time = df['Frequency (times)'].mean()
rerata_monet = df['Monetary (c.c. blood)'].mean()
rerata_month = df["Time (months)"].mean()

df = df.drop(columns=['whether he/she donated blood in March 2007'])

# print(data)

df = pd.merge(df, target, left_index=True,
              right_index=True, how='outer').dropna()

data_target = np.array(target)
data_target = data_target.astype(int)
data = np.array(df)  # konversi data csv menjadi array
data = np.array(df)
data = data.astype(int)
n_feature = len(data[0, :]) - 1
n_data = len(data[0, :])
print(n_data)

perceptron2 = Perceptron2(n_feature, length_data=n_data)
perceptron2.train_w(data)

# membaca jumlah feature
# n_feature = len(data_feature[0, :]) - 1

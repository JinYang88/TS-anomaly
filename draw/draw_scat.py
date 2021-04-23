import pandas as pd
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams['font.family'] = 'Times New Roman'
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['font.size'] = 16

model_color = {'KNN': 'black', 'iForest': 'blue', 'LODA': 'brown', 'LOF': 'gold', 'PCA': 'gray', 'AutoEncoder': 'green',
               'LSTM': 'orange', 'LSTM_VAE': 'pink', 'DAGMM': 'purple', 'MAD_GAN': 'red', 'MSCRED': 'violet',
               'OmniAnomaly': 'yellow'}
dataset_marker = {'SMD': ',', 'SMAP': 'o', 'MSL': 'v', 'WADI': 's', 'SWAT': 'h'}

salience_csv = pd.read_excel('./data/all_salience.xlsx').drop(index=0).reset_index(drop=True)
f1_csv = pd.read_excel('./data/all_f1.xlsx').drop(index=0).reset_index(drop=True)

dataset_list = salience_csv.columns.values[1:]
model_list = np.array(salience_csv['Models'])
salience_csv = salience_csv.drop(columns='Models')
f1_csv = f1_csv.drop(columns='Models')

plot = []

for i in range(len(model_list)):
    temp = []
    for j in range(len(dataset_list)):
        x = salience_csv.iloc[i, j]
        y = f1_csv.iloc[i, j]
        model = model_list[i]
        marker = dataset_list[j]
        temp.append(plt.scatter(x, y, s=30, c=model_color[model], marker=dataset_marker[marker]))
    plot.append(temp)

plot = np.array(plot)

l1 = plt.legend(plot[:, 0], model_list, loc='lower right')
l2 = plt.legend(plot[0, :], dataset_list, loc=8)
plt.gca().add_artist(l1)

plt.show()

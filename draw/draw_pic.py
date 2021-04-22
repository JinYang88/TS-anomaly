import pandas as pd
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt
rcParams['font.family'] = 'Times New Roman'
rcParams['font.sans-serif'] = ['Times New Roman']
rcParams['font.size'] = 16

raw_csv = pd.read_csv('./data/best_data(efficiency_only).csv')

data = raw_csv[raw_csv['dataset'] == 'WADI']

metrics = data[['model', 'train_time', 'test_time']]

model = np.array(metrics['model'])
train_time = np.array(metrics['train_time'])
test_time = np.array(metrics['test_time'])

width = 0.35       # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.bar(model, np.log(train_time), width, label='Train time')
ax.bar(model, np.log(test_time), width, bottom=np.log(train_time),
       label='Test Time')
ax.set_ylabel('Time/s')
ax.set_title('Efficiency of different models')
ax.legend()

plt.show()
ax.figure.savefig("./time_efficiency.pdf")

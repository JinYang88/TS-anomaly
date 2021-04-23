import pandas as pd
import numpy as np
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams["font.family"] = "Times New Roman"
rcParams["font.sans-serif"] = ["Times New Roman"]
rcParams["font.size"] = 16

raw_csv = pd.read_csv("./data/best_data(efficiency_only).csv")

data = raw_csv[raw_csv["dataset"] == "WADI"]

metrics = data[["model", "train_time", "test_time"]]

model = np.array(metrics["model"])
train_time = np.array(metrics["train_time"])
test_time = np.array(metrics["test_time"])

width = 0.35  # the width of the bars: can also be len(x) sequence

fig, ax = plt.subplots()

ax.set_yscale("symlog")
x = np.arange(len(model))
plt.xticks(x, model)
ax.bar(x - width / 2, train_time, width, label="Train time")
ax.bar(x + width / 2, test_time, width, label="Test Time")


def convert_unit(data_list):
    convert = []
    for i in range(len(data_list)):
        if data_list[i] > 60000:
            convert.append(str(int(data_list[i] / 3600)) + "h")
        elif 1000 < data_list[i] < 60000:
            convert.append(str(int(data_list[i] / 60)) + "m")
        else:
            convert.append(str(int(data_list[i])) + "s")

    return convert


train_convert = convert_unit(train_time)
test_convert = convert_unit(test_time)

train_value = [round(i, 3) for i in train_time]
for i in range(len(train_value)):
    print(train_convert[i])
    plt.text(x=i - width, y=train_value[i], s=train_convert[i], size=15)

test_value = [round(i, 3) for i in test_time]
for i in range(len(test_value)):
    plt.text(x=i, y=test_value[i], s=test_convert[i], size=15)


ax.set_ylabel("Time (s)")
# ax.set_title('Efficiency of different models')
ax.legend()

plt.show()
ax.figure.savefig('./time_efficiency.pdf')


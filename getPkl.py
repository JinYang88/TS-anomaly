import pickle
import pandas as pd
import numpy as np

data = np.array(pd.read_csv("./datasets/anomaly/SMD/test_label/machine-1-1.csv", header=None))

f = open("./datasets/anomaly/SMD/test_label/machine-1-1_test_label.pkl", 'wb')
pickle.dump(data, f)


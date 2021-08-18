import numpy as np
from IPython import embed

class ThreeSigma:
    def __init__(self, anomaly_ts_num=0.5):
        self.anomaly_ts_num = anomaly_ts_num
        self.lower_threshold_list = []
        self.upper_threshold_list = []

    def fit(self,train_data):
        for i in range(train_data.shape[1]):
            ymean = np.mean(train_data[:, i])
            ystd = np.std(train_data[:, i])
            threshold1 = ymean - 3 * ystd
            threshold2 = ymean + 3 * ystd
            self.lower_threshold_list.append(threshold1)
            self.upper_threshold_list.append(threshold2)

    def predict(self,test_data):
        anomaly_indice = np.zeros(test_data.shape[0])
        for dim in range(test_data.shape[1]):
            ts = test_data[:, dim]
            lower = self.lower_threshold_list[dim]
            upper = self.upper_threshold_list[dim]
            anomaly_dim_indice = (ts < lower).astype(int) | (ts > upper).astype(int)
            anomaly_indice += anomaly_dim_indice
        anomaly = (anomaly_indice > int(self.anomaly_ts_num * test_data.shape[1])).astype(bool)
        return anomaly
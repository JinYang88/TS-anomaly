import numpy as np
from pyod.models.loda import LODA
from IPython import embed


class LODAUni:
    def __init__(self, anomaly_ts_num: float = 0.5, anomaly_threshold=0.2):
        self.anomaly_ts_num = anomaly_ts_num
        self.anomaly_threshold = anomaly_threshold
        self.LODAModels = []

    def fit(self, train_data, n_bins=10):
        """
        train_data: type is np.ndarray
        """
        for i in range(train_data.shape[1]):
            print(f"fit dim {i}")
            ts = train_data[:, i].copy()
            od = LODA(n_bins=n_bins)
            od.fit(ts.reshape(-1, 1))
            self.LODAModels.append(od)

    def predict(self, test_data):
        """
        test_data: type is np.ndarray
        """
        anomaly_indice = np.zeros(test_data.shape[0])
        for dim in range(test_data.shape[1]):
            print(f"predict dim {dim}")
            ts = test_data[:, dim].copy()
            anomaly_score = self.LODAModels[dim].decision_function(
                ts.reshape(-1, 1))
            anomaly_dim_indice = (
                anomaly_score > self.anomaly_threshold).astype(int)
            anomaly_indice += anomaly_dim_indice
        anomaly = (anomaly_indice > int(
            self.anomaly_ts_num * test_data.shape[1])).astype(bool)
        return anomaly

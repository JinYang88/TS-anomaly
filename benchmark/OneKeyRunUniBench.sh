DATASET="SMD"
THRES=0.5

python 11_MA_benchmark.py --dataset $DATASET --anomaly_ts_num $THRES
python 12_AR_benchmark.py --dataset $DATASET --anomaly_ts_num $THRES
python 13_ARIMA_benchmark.py --dataset $DATASET --anomaly_ts_num $THRES
python 14_iforest_univariate_benchmark.py --dataset $DATASET --anomaly_ts_num $THRES --n_estimators 10 --anomaly_threshold 0.02 --anomaly_ts_num
python 15_LODA_uni_benchmark.py --dataset $DATASET --anomaly_ts_num $THRES --n_bins 2 --anomaly_threshold 0.2
python 16_ocsvm_uni_benchmark.py --dataset $DATASET --anomaly_ts_num $THRES --anomaly_threshold 10

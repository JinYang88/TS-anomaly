import os
import pickle


# because the length of SWAT and WADI is very large
def split_data(subdataset_num, dataset_name):

    data_path = "./" + dataset_name + "/processed/"
    save_root_path = "./" + dataset_name + "_SPLIT" + "/processed/"
    os.makedirs(save_root_path, exist_ok=True)

    train_path = data_path + dataset_name.lower() + "_train.pkl"
    test_path = data_path + dataset_name.lower() + "_test.pkl"
    test_label_path = data_path + dataset_name.lower() + "_test_label.pkl"

    train_data = pickle.load(open(train_path, "rb"))
    train_length = train_data.shape[0]
    train_sub_size = train_length // subdataset_num
    for i in range(0, subdataset_num):
        save_path = (
            save_root_path + dataset_name.lower() + "-" + str(i + 1) + "_train.pkl"
        )
        if i == subdataset_num - 1:
            sub_dataset = train_data[i * train_sub_size :, :]
        else:
            sub_dataset = train_data[i * train_sub_size : (i + 1) * train_sub_size, :]
        pickle.dump(sub_dataset, open(save_path, "wb"))

    test_data = pickle.load(open(test_path, "rb"))
    test_length = test_data.shape[0]
    test_sub_size = test_length // subdataset_num
    for i in range(0, subdataset_num):
        save_path = (
            save_root_path + dataset_name.lower() + "-" + str(i + 1) + "_test.pkl"
        )
        if i == subdataset_num - 1:
            sub_dataset = test_data[i * test_sub_size :, :]
        else:
            sub_dataset = test_data[i * test_sub_size : (i + 1) * test_sub_size, :]
        pickle.dump(sub_dataset, open(save_path, "wb"))

    test_label_data = pickle.load(open(test_label_path, "rb"))
    test_label_length = test_label_data.shape[0]
    test_label_sub_size = test_label_length // subdataset_num
    for i in range(0, subdataset_num):
        save_path = (
            save_root_path + dataset_name.lower() + "-" + str(i + 1) + "_test_label.pkl"
        )
        if i == subdataset_num - 1:
            sub_dataset = test_label_data[i * test_label_sub_size :]
        else:
            sub_dataset = test_label_data[
                i * test_label_sub_size : (i + 1) * test_label_sub_size
            ]
        pickle.dump(sub_dataset, open(save_path, "wb"))

        anomaly_ratio = list(sub_dataset).count(1) / sub_dataset.shape[0]
        print(anomaly_ratio)


split_data(2, "SWAT")

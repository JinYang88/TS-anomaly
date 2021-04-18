import os
import json
import pandas as pd

root_dir = './'

subdataset_num = {'SMD': 28, 'SMAP': 54, 'MSL': 27, 'SWAT': 1, 'WADI': 1,
                  'SWAT_SPLIT': 3, 'WADI_SPLIT': 3}

metric_key_num = 10


def get_dir_list(path):
    dir_list = []
    for item in os.listdir(path):
        directory = os.path.join(path, item).replace('\\', '/')
        if os.path.isdir(directory):
            dir_list.append(item)

    return dir_list


def refine_df(dataFrame):
    dataFrame = pd.DataFrame(dataFrame.groupby([dataFrame.model, dataFrame.dataset]).agg({'adj_f1': 'max'}, 'hash_id'))
    print(dataFrame)


data = pd.DataFrame(columns=['model', 'dataset', 'hash_id', 'adj_f1', 'raw_f1', 'time'])
res_dict = dict()

for model in get_dir_list(root_dir):
    model_dirs = root_dir + model
    for hash_id in get_dir_list(model_dirs):
        hash_id_dirs = model_dirs + '/' + hash_id
        for dataset in get_dir_list(hash_id_dirs):
            dataset_dirs = hash_id_dirs + '/' + dataset
            subdataset_count = 0
            miss_metrics = False
            miss_time = False
            miss_key = False
            avg_adj_f1 = 0
            avg_raw_f1 = 0
            time = 0
            for subdataset in get_dir_list(dataset_dirs):
                subdataset_dir = dataset_dirs + '/' + subdataset
                subdataset_count += 1
                metrics_dir = subdataset_dir + '/' + 'metrics.json'
                time_dir = subdataset_dir + '/' + 'time.json'
                if not os.path.exists(time_dir):
                    miss_time = True
                else:
                    if not os.path.exists(metrics_dir):
                        miss_metrics = True
                    else:

                        metrics_dict = json.load(open(metrics_dir))
                        time_dict = json.load(open(time_dir))
                        if len(metrics_dict) != metric_key_num:
                            miss_key = True
                        else:
                            avg_adj_f1 += metrics_dict['adj_f1']
                            avg_raw_f1 += metrics_dict['raw_f1']
                            time += time_dict['train'] + time_dict['test']

            avg_adj_f1 = avg_adj_f1 / subdataset_num[dataset]
            avg_raw_f1 = avg_raw_f1 / subdataset_num[dataset]

            if subdataset_count == subdataset_num[dataset] and (not miss_time) and (not miss_metrics)\
                    and (not miss_key):
                data.loc[len(data)] = [model, dataset, hash_id, avg_adj_f1, avg_raw_f1, time]
                # data_dict.update({(model, dataset, hash_id): (avg_adj_f1, avg_raw_f1, time)})
            elif miss_time:
                print('Time json missing:' + model + '/' + dataset + '/' + hash_id)
            elif miss_metrics:
                print('Metrics json missing:' + model + '/' + dataset + '/' + hash_id)
            elif miss_key:
                print('Metrics key missing:' + model + '/' + dataset + '/' + hash_id)
            else:
                print('Subdataset missing:' + model + '/' + dataset + '/' + hash_id)

print(data)
print(refine_df(data))

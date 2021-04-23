from common.evaluation import compute_delay
import pickle
import pandas as pd
import numpy as np
import os

csv_data = pd.read_csv('./best_data(efficiency_only).csv')
models = csv_data['model'].unique()
datasets = csv_data['dataset'].unique()


def get_hash(data, mod, dset):
    data = pd.DataFrame(data)
    data = data[data['model'] == mod]
    data = data[data['dataset'] == dset]

    hash = np.array(data['hash_id'])[0]

    return hash


def compute_d(th, sc):
    delay = 0
    for subdataset, t in th.items():
        s = sc[subdataset]['test_score']
        label = sc[subdataset]['label']
        pred = np.array(s > t).astype(int)
        delay += compute_delay(label, pred)

    delay /= len(th)

    return delay


with open('./detailed_dict.pkl', 'rb') as f1:
    threds = pickle.load(f1)
with open('./all_best_scores.pkl', 'rb') as f2:
    scores = pickle.load(f2)

for dataset in datasets:
    csv = pd.DataFrame(columns=['model', 'a_delay', 'r_delay', 'p_delay'])
    for model in models:
        hash_id = get_hash(csv_data, model, dataset)

        thred = threds[(hash_id, model)]
        score = scores[(hash_id, model)]

        a_thred = thred['a_theta']
        r_thred = thred['r_theta']
        p_thred = thred['p_theta']

        a_delay = compute_d(a_thred, score)
        r_delay = compute_d(r_thred, score)
        p_delay = compute_d(p_thred, score)

        csv.loc[len(csv)] = [model, a_delay, r_delay, p_delay]

    if not os.path.exists('./delay'):
        os.mkdir('./delay')

    csv.to_csv('./delay/' + dataset + '.csv')



'''
label = updated_res_dict[('789dc9fd', 'AutoEncoder')]["M-5"]["label"]
score = updated_res_dict[('789dc9fd', 'AutoEncoder')]["M-5"]["test_score"]
pred = (score > 0.4040404040404041).astype(int)
# pred

delay = compute_delay(label, pred)

with open("./compute_delay/detailed_dict.pkl", "rb") as fr:
    threds = pickle.load(fr) #阈值

# ["a_theta", "p_theta", "r_theta"]
threds[('789dc9fd', 'AutoEncoder')]["a_theta"]["M-5"], threds[('789dc9fd', 'AutoEncoder')]["r_theta"]["M-5"], threds[('789dc9fd', 'AutoEncoder')]["p_theta"]["M-5"]

# 为每个dataset生成一个文件，每个文件包含所有方法的三个delay
'''

from common.dataloader import load_CSV_dataset, load_dataset
import os
import pickle
from IPython import embed
import numpy as np

target_line_num = 1944720

dataset = "SMD"
subdataset = "machine-1-1"
data_dict = load_dataset(dataset, subdataset)
num = data_dict["train"].shape[0]
factor = target_line_num // num
add = target_line_num % num

save_dir = "./datasets/anomaly/SMD/oversampled"
os.makedirs(save_dir, exist_ok=True)


for dtype in ["train", "test", "test_labels"]:
    data = data_dict[dtype]
    final_data = []
    for i in range(factor):
        final_data.append(data)
    final_data.append(data[0:add])
    final_data = np.concatenate(final_data)

    with open(
        os.path.join(save_dir, "{}_{}.pkl".format(subdataset, dtype)), "wb"
    ) as fw:
        print("Dumping {}, with shape {}".format(dtype, final_data.shape))
        pickle.dump(final_data, fw)

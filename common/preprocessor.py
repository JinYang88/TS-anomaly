import torch
from common.sliding import BatchSlidingWindow

def preprocess_SMD(data_dict, window_size=100):
    train_windows = []
    test_windows = []
    
    for data_name, sub_dict in data_dict.items():
        if not isinstance(sub_dict, dict): continue
        train = sub_dict["train"]
        test = sub_dict["test"]
        train_win = BatchSlidingWindow(train.shape[0], window_size=window_size, batch_size=1000, shuffle=False).get_windows(train)
        test_win = BatchSlidingWindow(test.shape[0], window_size=window_size, batch_size=1000, shuffle=False).get_windows(test)

        train_windows.append(train_win)
        test_windows.append(test_win)
    
    return torch.cat(train_windows,dim=0), torch.cat(test_windows,dim=0)
        
        

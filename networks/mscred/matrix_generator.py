import os
import numpy as np


def generate_signature_matrix_node(data_dict, matrix_path, save_dir, gap_time, win_size):
    data = np.concatenate((data_dict["train"], data_dict["test"]), axis=0)
    data = data.transpose()
    sensor_n = data.shape[0]
    length = data.shape[1]
    # min-max normalization
    max_value = np.max(data, axis=1)
    min_value = np.min(data, axis=1)
    data = (np.transpose(data) - min_value) / (max_value - min_value + 1e-6)
    data = np.transpose(data)

    # multi-scale signature matrix generation
    for w in range(len(win_size)):
        matrix_all = []
        win = win_size[w]
        for t in range(0, length, gap_time):
            # print t
            matrix_t = np.zeros((sensor_n, sensor_n))
            if t >= max(win_size):
                for l in range(sensor_n):
                    for m in range(l, sensor_n):
                        # if np.var(data[i, t - win:t]) and np.var(data[j, t - win:t]):
                        matrix_t[l][m] = np.inner(data[l, t - win:t], data[m, t - win:t]) / win  # rescale by win
                        matrix_t[m][l] = matrix_t[l][m]
            matrix_all.append(matrix_t)

        matrix_data_path = save_dir + "matrix_data_" + matrix_path + '/'

        if not os.path.exists(matrix_data_path):
            os.makedirs(matrix_data_path)
        path_temp = matrix_data_path + "matrix_win_" + str(win)
        np.save(path_temp, matrix_all)
        del matrix_all[:]
    print('Generation for ' + matrix_path + ' complete')


def generate_train_test_data(matrix_path, x_train, x_test, save_dir, step_max, gap_time, win_size):
    # data sample generation
    print("generating train/test data samples of " + matrix_path)
    matrix_data_path = save_dir + "matrix_data_" + matrix_path + '/'

    train_start = 0
    train_end = x_train.shape[0]
    test_start = train_end
    test_end = x_train.shape[0] + x_test.shape[0]

    train_data_path = matrix_data_path + "train_data/"
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)
    test_data_path = matrix_data_path + "test_data/"
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    data_all = []

    for w in range(len(win_size)):
        path_temp = matrix_data_path + "matrix_win_" + str(win_size[w]) + ".npy"
        data_all.append(np.load(path_temp))

    train_test_time = [[train_start, train_end], [test_start, test_end]]
    for m in range(len(train_test_time)):
        for data_id in range(int(train_test_time[m][0] / gap_time), int(train_test_time[m][1] / gap_time)):
            # print data_id
            step_multi_matrix = []
            for step_id in range(step_max, 0, -1):
                multi_matrix = []
                for k in range(len(win_size)):
                    multi_matrix.append(data_all[k][data_id - step_id])
                step_multi_matrix.append(multi_matrix)

            if (train_start / gap_time + win_size[-1] / gap_time + step_max) <= data_id < (
                    train_end / gap_time):  # remove start points with invalid value
                path_temp = os.path.join(train_data_path, 'train_data_' + str(data_id))
                np.save(path_temp, step_multi_matrix)
            elif (test_start / gap_time) <= data_id < (test_end / gap_time):
                path_temp = os.path.join(test_data_path, 'test_data_' + str(data_id))
                np.save(path_temp, step_multi_matrix)

            del step_multi_matrix[:]
    print("train/test data generation finish!")

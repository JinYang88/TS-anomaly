import time
import torch
import shutil
import os
import torch.nn as nn
import numpy as np
from .convolution_lstm import ConvLSTM
from .matrix_generator import generate_signature_matrix_node, generate_train_test_data
from .utils import train, test, load_signature_data, evaluate
from common.utils import set_device


def attention(ConvLstm_out):
    attention_w = []
    for k in range(5):
        attention_w.append(torch.sum(torch.mul(ConvLstm_out[k], ConvLstm_out[-1])) / 5)
    m = nn.Softmax(dim=0)
    attention_w = torch.reshape(m(torch.stack(attention_w)), (-1, 5))
    cl_out_shape = ConvLstm_out.shape
    ConvLstm_out = torch.reshape(ConvLstm_out, (5, -1))
    convLstmOut = torch.matmul(attention_w, ConvLstm_out)
    convLstmOut = torch.reshape(
        convLstmOut, (cl_out_shape[1], cl_out_shape[2], cl_out_shape[3])
    )
    return convLstmOut


class CnnEncoder(nn.Module):
    def __init__(self, in_channels_encoder):
        super(CnnEncoder, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels_encoder, 32, 3, (1, 1), 1), nn.SELU()
        )
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 2, (2, 2), 0), nn.SELU())
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, 2, (2, 2), 0), nn.SELU())
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, 2, (2, 2), 0), nn.SELU())

    def forward(self, X):
        conv1_out = self.conv1(X)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        return conv1_out, conv2_out, conv3_out, conv4_out


class Conv_LSTM(nn.Module):
    def __init__(self, device):
        super(Conv_LSTM, self).__init__()
        self.conv1_lstm = ConvLSTM(
            input_channels=32,
            hidden_channels=[32],
            kernel_size=3,
            step=5,
            effective_step=[4],
            device=device,
        )
        self.conv2_lstm = ConvLSTM(
            input_channels=64,
            hidden_channels=[64],
            kernel_size=3,
            step=5,
            effective_step=[4],
            device=device,
        )
        self.conv3_lstm = ConvLSTM(
            input_channels=128,
            hidden_channels=[128],
            kernel_size=3,
            step=5,
            effective_step=[4],
            device=device,
        )
        self.conv4_lstm = ConvLSTM(
            input_channels=256,
            hidden_channels=[256],
            kernel_size=3,
            step=5,
            effective_step=[4],
            device=device,
        )

    def forward(self, conv1_out, conv2_out, conv3_out, conv4_out):
        conv1_lstm_out = self.conv1_lstm(conv1_out)
        conv1_lstm_out = attention(conv1_lstm_out[0][0])
        conv2_lstm_out = self.conv2_lstm(conv2_out)
        conv2_lstm_out = attention(conv2_lstm_out[0][0])
        conv3_lstm_out = self.conv3_lstm(conv3_out)
        conv3_lstm_out = attention(conv3_lstm_out[0][0])
        conv4_lstm_out = self.conv4_lstm(conv4_out)
        conv4_lstm_out = attention(conv4_lstm_out[0][0])
        return (
            conv1_lstm_out.unsqueeze(0),
            conv2_lstm_out.unsqueeze(0),
            conv3_lstm_out.unsqueeze(0),
            conv4_lstm_out.unsqueeze(0),
        )


class CnnDecoder(nn.Module):
    def __init__(self, in_channels):
        super(CnnDecoder, self).__init__()
        self.deconv4_a = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, 2, 2, 0, 0), nn.SELU()
        )
        self.deconv4_b = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 128, 2, 2, 0, 1), nn.SELU()
        )
        self.deconv3_a = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, 2, 0, 0), nn.SELU()
        )
        self.deconv3_b = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 2, 2, 0, 1), nn.SELU()
        )
        self.deconv2_a = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 2, 2, 0, 0), nn.SELU()
        )
        self.deconv2_b = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 2, 2, 0, 1), nn.SELU()
        )
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(64, 3, 3, 1, 1, 0), nn.SELU())

    def forward(self, conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out):
        if conv3_lstm_out.shape[-1] % 2 == 0:
            deconv4 = self.deconv4_a(conv4_lstm_out)
        else:
            deconv4 = self.deconv4_b(conv4_lstm_out)
        deconv4_concat = torch.cat((deconv4, conv3_lstm_out), dim=1)
        if conv2_lstm_out.shape[-1] % 2 == 0:
            deconv3 = self.deconv3_a(deconv4_concat)
        else:
            deconv3 = self.deconv3_b(deconv4_concat)
        deconv3_concat = torch.cat((deconv3, conv2_lstm_out), dim=1)
        if conv1_lstm_out.shape[-1] % 2 == 0:
            deconv2 = self.deconv2_a(deconv3_concat)
        else:
            deconv2 = self.deconv2_b(deconv3_concat)
        deconv2_concat = torch.cat((deconv2, conv1_lstm_out), dim=1)
        deconv1 = self.deconv1(deconv2_concat)
        return deconv1


class MSCRED(nn.Module):
    def __init__(
        self,
        in_channels_encoder,
        in_channels_decoder,
        save_path,
        device,
        step_max,
        gap_time,
        win_size,
        learning_rate,
        epoch,
        thred_b,
    ):
        super(MSCRED, self).__init__()
        self.device = set_device(device)
        self.cnn_encoder = CnnEncoder(in_channels_encoder)
        self.conv_lstm = Conv_LSTM(self.device)
        self.cnn_decoder = CnnDecoder(in_channels_decoder)
        self.save_path = save_path
        self.step_max = step_max
        self.gap_time = gap_time
        self.win_size = win_size
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.thred_b = thred_b
        self.time_tracker = {}

        if os.path.exists(os.path.dirname(self.save_path)):
            shutil.rmtree(os.path.dirname(self.save_path))

    def forward(self, x):
        conv1_out, conv2_out, conv3_out, conv4_out = self.cnn_encoder(x)
        conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out = self.conv_lstm(
            conv1_out, conv2_out, conv3_out, conv4_out
        )

        gen_x = self.cnn_decoder(
            conv1_lstm_out, conv2_lstm_out, conv3_lstm_out, conv4_lstm_out
        )
        return gen_x

    def data_preprocessing(self, data_dict):
        generate_signature_matrix_node(
            data_dict,
            self.save_path,
            self.gap_time,
            self.win_size,
        )

        x_train = data_dict["train"]
        x_test = data_dict["test"]

        generate_train_test_data(
            x_train,
            x_test,
            self.save_path,
            self.step_max,
            self.gap_time,
            self.win_size,
        )

    def fit(self, data_dict):
        start = time.time()
        self.data_preprocessing(data_dict)
        signature_data_dict = load_signature_data(self.save_path)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        train(
            signature_data_dict["train"],
            self,
            optimizer,
            epochs=self.epoch,
            device=self.device,
        )

        end = time.time()
        self.time_tracker["train"] = end - start

    def predict_prob(self, len_x_train, x_test, x_test_labels=None):
        signature_data_dict = load_signature_data(self.save_path)
        start = time.time()
        test(
            signature_data_dict["test"],
            self,
            len_x_train,
            save_dir=self.save_path,
            gap_time=self.gap_time,
            device=self.device,
        )
        end = time.time()
        self.time_tracker["test"] = end - start

        anomaly_score = evaluate(self.save_path, self.thred_b, self.gap_time)

        if x_test_labels is not None:
            anomaly_label = x_test_labels[self.gap_time - 1 :]
            length = min(len(anomaly_score), len(anomaly_label))
            anomaly_score = anomaly_score[0:length]
            anomaly_label = anomaly_label[0:length]
            return np.array(anomaly_score), np.array(anomaly_label)
        else:
            return np.array(anomaly_score)

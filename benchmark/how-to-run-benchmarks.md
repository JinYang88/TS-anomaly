下面这几个模型需要tf1.12相关的依赖，需要新建一个**py3.6（不能更高）**的conda env，再

```pip install -r tf_requirements.txt```

然后就能run下面的了


```
python 6_lstm_vae_benchmark.py --dataset HUAWEI_GROUP_C --lr 0.001 --z_dim 3 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 10 B
python 7_dagmm_benchmark.py --dataset HUAWEI_GROUP_C  -ch 32 16 2 -eh 80 40 B
python 8_omnianomaly_benchmark.py --dataset HUAWEI_GROUP_C --lr 0.001 --z_dim 3 --rnn_num_hidden 500 --window_size 32 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 2 B

python 8_omnianomaly_benchmark.py --dataset HUAWEI_GROUP_C --lr 0.001 --z_dim 3 --rnn_num_hidden 500 --window_size 64 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 1 B
python 8_omnianomaly_benchmark.py --dataset HUAWEI_GROUP_C --lr 0.001 --z_dim 3 --rnn_num_hidden 500 --window_size 16 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 1 B
python 8_omnianomaly_benchmark.py --dataset HUAWEI_GROUP_C --lr 0.001 --z_dim 3 --rnn_num_hidden 500 --window_size 128 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 1 B
python 8_omnianomaly_benchmark.py --dataset HUAWEI_GROUP_C --lr 0.001 --z_dim 3 --rnn_num_hidden 500 --window_size 16 --stride 1 --dense_dim 128 --nf_layers 2 --max_epoch 1 B
python 8_omnianomaly_benchmark.py --dataset HUAWEI_GROUP_C --lr 0.001 --z_dim 3 --rnn_num_hidden 500 --window_size 16 --stride 2 --dense_dim 128 --nf_layers 2 --max_epoch 1 B


python 7_dagmm_benchmark.py --dataset HUAWEI_GROUP_B  -ch 32 16 2 -eh 60 30 B
python 7_dagmm_benchmark.py --dataset HUAWEI_GROUP_B  -ch 32 16 8 -eh 80 40 B
```


下面的模型依赖不复杂，最好用一个和上面的模型分开的conda env，缺啥装啥就可以了

```
python 1_iforest_benchmark.py --dataset HUAWEI_GROUP_C --n_estimators 100 B
python 2_LODA_benchmark.py --dataset HUAWEI_GROUP_C B
python 3_PCA_benchmark.py --dataset HUAWEI_GROUP_C --n_components 5 B
python 11_OCSVM_benchmark.py --dataset HUAWEI_GROUP_C B
python 4_AutoEncoder_benchmark.py --dataset HUAWEI_GROUP_C --hidden_neurons 64 5 5 64 --batch_size 32 --epochs 100 --l2_regularizer 0.1 B
python 5_lstm_benchmark.py --dataset HUAWEI_GROUP_C --lr 0.001 --window_size 32 --stride 5 --num_layers 2 --hidden_size 64 --gpu -1 B
python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_C --lr 0.001 --window_size 64 --stride 5 --gpu 0 B

python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --window_size 32 --stride 1 --gpu 0 B
python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --window_size 16 --stride 1 --gpu 0 B
python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --window_size 8 --stride 1 --gpu 0 B
python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --window_size 32 --stride 5 --gpu 0 B
python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --window_size 16 --stride 5 --gpu 0 B

python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --window_size 16 --stride 10 --gpu 0 B
python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --window_size 16 --stride 3 --gpu 0 B
python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --window_size 16 --stride 15 --gpu 0 B
python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.01 --inter FM_com --window_size 16 --stride 5 --gpu 0 B

python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.01 --window_size 8 --stride 5 --inter FM_com --gpu 0 B
python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.01 --window_size 5 --stride 5 --inter FM_com --gpu 0 B

10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.01 --window_size 8 --stride 5 --gpu 0 B


python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --window_size 64 --stride 1 --gpu 0 B
python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --window_size 128 --stride 1 --gpu 0 B
python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --window_size 32 --stride 2 --gpu 0 B
python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --window_size 32 --stride 8 --gpu 0 B


python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --gamma 0.1 --window_size 64 --stride 1 --gpu 0 B
python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --gamma 0.1 --window_size 128 --stride 1 --gpu 0 B
python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --gamma 0.1 --window_size 32 --stride 2 --gpu 0 B
python 10_cmanomaly_old_benchmark.py --dataset HUAWEI_GROUP_B --lr 0.001 --inter FM_com --gamma 0.1 --window_size 32 --stride 8 --gpu 0 B

```

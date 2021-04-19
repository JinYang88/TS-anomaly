python lstm_vae_benchmark.py --dataset MSL --lr 0.001 --z_dim 2 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 40
python lstm_vae_benchmark.py --dataset SMAP --lr 0.001 --z_dim 2 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 40
python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 3 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 10
python lstm_vae_benchmark.py --dataset WADI --lr 0.001 --z_dim 3 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 40
python omnianomaly_benchmark.py --dataset MSL --lr 0.001 --z_dim 3 --rnn_num_hidden 300 --window_size 64 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 10
python omnianomaly_benchmark.py --dataset SMAP --lr 0.001 --z_dim 3 --rnn_num_hidden 300 --window_size 64 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 10
python omnianomaly_benchmark.py --dataset SMD --lr 0.001 --z_dim 3 --rnn_num_hidden 300 --window_size 64 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 10
python omnianomaly_benchmark.py --dataset SWAT --lr 0.001 --z_dim 3 --rnn_num_hidden 200 --window_size 64 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 10
python omnianomaly_benchmark.py --dataset WADI --lr 0.001 --z_dim 3 --rnn_num_hidden 500 --window_size 32 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 10

export PATH=$PATH:/research/dept7/jyliu/cuda/cuda-9.0/bin
export LD_LIBRARY_PATH=/research/dept7/jyliu/cuda/cuda-9.0/lib64


# python lstm_vae_benchmark.py --dataset MSL --lr 0.001 --z_dim 2 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 40
# python lstm_vae_benchmark.py --dataset SMAP --lr 0.001 --z_dim 2 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 40
# python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 3 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 10
# python lstm_vae_benchmark.py --dataset WADI --lr 0.001 --z_dim 3 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 40

# running on omni
CUDA_VISIBLE_DEVICES="" python lstm_vae_benchmark.py --dataset SWAT --lr 0.001 --z_dim 3 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 40 > logs/lstm_vae_swat_best.log 2>&1 &



# python dagmm_benchmark.py --dataset SMD --lr 0.0003 --dropout 0.25 --num_epochs 100 -ch 128 64 2 -eh 100 50
# python dagmm_benchmark.py --dataset SWAT --lr 0.0001 --dropout 0.25 --num_epochs 100 -ch 64 32 2 -eh 60 30
# python dagmm_benchmark.py --dataset WADI --lr 0.0001 --dropout 0.25 --num_epochs 100 -ch 128 64 2 -eh 100 50



# running
( CUDA_VISIBLE_DEVICES=0 python omnianomaly_benchmark.py --dataset SWAT --lr 0.001 --z_dim 3 --rnn_num_hidden 200 --window_size 64 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 10 && CUDA_VISIBLE_DEVICES=0 python omnianomaly_benchmark.py --dataset WADI --lr 0.001 --z_dim 3 --rnn_num_hidden 500 --window_size 32 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 10 && CUDA_VISIBLE_DEVICES=0 python omnianomaly_benchmark.py --dataset MSL --lr 0.001 --z_dim 3 --rnn_num_hidden 300 --window_size 64 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 10 && CUDA_VISIBLE_DEVICES=0 python omnianomaly_benchmark.py --dataset SMAP --lr 0.001 --z_dim 3 --rnn_num_hidden 300 --window_size 64 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 10 ) > logs/omni_all.log 2>&1 &



# CUDA_VISIBLE_DEVICES=2 python omnianomaly_benchmark.py --dataset SMD --lr 0.001 --z_dim 3 --rnn_num_hidden 300 --window_size 64 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 10 > logs/omni_msl.log 2>&1 &

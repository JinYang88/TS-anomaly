# sh benchmarking_scripts/SMD/lstm_vae.sh > logs/lstm_vae.SMD.bench 2>&1 &

export PATH=$PATH:/research/dept7/jyliu/cuda/cuda-9.0/bin
export LD_LIBRARY_PATH=/research/dept7/jyliu/cuda/cuda-9.0/lib64


# z_dim 1-4
python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 1 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 40

python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 2 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 40

python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 3 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 40

python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 4 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 40

# window_size, 64

python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 1 --intermediate_dim 64 --window_size 64 --stride 5 --hidden_size 128 --num_epochs 40

python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 2 --intermediate_dim 64 --window_size 64 --stride 5 --hidden_size 128 --num_epochs 40

python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 3 --intermediate_dim 64 --window_size 64 --stride 5 --hidden_size 128 --num_epochs 40

python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 4 --intermediate_dim 64 --window_size 64 --stride 5 --hidden_size 128 --num_epochs 40

# window_size, 128

python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 1 --intermediate_dim 64 --window_size 128 --stride 5 --hidden_size 128 --num_epochs 40

python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 2 --intermediate_dim 64 --window_size 128 --stride 5 --hidden_size 128 --num_epochs 40

python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 3 --intermediate_dim 64 --window_size 128 --stride 5 --hidden_size 128 --num_epochs 40

python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 4 --intermediate_dim 64 --window_size 128 --stride 5 --hidden_size 128 --num_epochs 40


# intermediate_dim 32

python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 1 --intermediate_dim 32 --window_size 64 --stride 5 --hidden_size 128 --num_epochs 40

python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 2 --intermediate_dim 32 --window_size 64 --stride 5 --hidden_size 128 --num_epochs 40

python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 3 --intermediate_dim 32 --window_size 64 --stride 5 --hidden_size 128 --num_epochs 40

python lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 4 --intermediate_dim 32 --window_size 64 --stride 5 --hidden_size 128 --num_epochs 40

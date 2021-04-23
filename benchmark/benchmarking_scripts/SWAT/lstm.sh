# sh benchmarking_scripts/SWAT/lstm.sh > logs/lstm.SWAT.bench 2>&1 &

python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 32 --stride 5 --num_layers 1 --hidden_size 32 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 32 --stride 5 --num_layers 1 --hidden_size 64 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 32 --stride 5 --num_layers 1 --hidden_size 128 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 32 --stride 5 --num_layers 1 --hidden_size 256 --gpu 7

python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 32 --stride 5 --num_layers 2 --hidden_size 32 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 32 --stride 5 --num_layers 2 --hidden_size 64 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 32 --stride 5 --num_layers 2 --hidden_size 128 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 32 --stride 5 --num_layers 2 --hidden_size 256 --gpu 7


python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 3 --hidden_size 32 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 3 --hidden_size 64 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 3 --hidden_size 128 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 3 --hidden_size 256 --gpu 7


python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 1 --hidden_size 32 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 1 --hidden_size 64 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 1 --hidden_size 128 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 1 --hidden_size 256 --gpu 7

python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 2 --hidden_size 32 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 2 --hidden_size 64 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 2 --hidden_size 128 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 2 --hidden_size 256 --gpu 7


python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 3 --hidden_size 32 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 3 --hidden_size 64 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 3 --hidden_size 128 --gpu 7
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 64 --stride 5 --num_layers 3 --hidden_size 256 --gpu 7

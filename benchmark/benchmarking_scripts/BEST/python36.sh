python mscred_benchmark.py --dataset MSL --lr 0.003 --in_channels_encoder 3 --in_channels_decoder 256 --num_epochs 1 --gpu 0
python mscred_benchmark.py --dataset SMAP --lr 0.003 --in_channels_encoder 3 --in_channels_decoder 256 --num_epochs 1 --gpu 0
python mscred_benchmark.py --dataset SWAT --lr 0.003 --in_channels_encoder 3 --in_channels_decoder 256 --hidden_size 64 --num_epochs 1 --gpu 0
python mscred_benchmark.py --dataset WADI_SPLIT --lr 0.003 --in_channels_encoder 3 --in_channels_decoder 256 --hidden_size 64 --num_epochs 1 --gpu 0

python mad_gan_benchmark.py --dataset MSL --window_size 16 --stride 1 --lr 0.001 --num_epochs 150
python mad_gan_benchmark.py --dataset SMAP --window_size 16 --stride 1 --lr 0.005 --num_epochs 50
python mad_gan_benchmark.py --dataset SMD --window_size 32 --stride 5 --lr 0.001 --num_epochs 50
python mad_gan_benchmark.py --dataset SWAT --window_size 64 --stride 5 --lr 0.005 --num_epochs 50
python mad_gan_benchmark.py --dataset WADI --window_size 64 --stride 5 --lr 0.005 --num_epochs 50

python AutoEncoder_benchmark.py --dataset MSL --hidden_neurons 16 8 8 16 --batch_size 512 --epochs 100 --l2_regularizer 0.00001
python AutoEncoder_benchmark.py --dataset SMAP --hidden_neurons 32 16 16 32 --batch_size 512 --epochs 100 --l2_regularizer 0.00001
python AutoEncoder_benchmark.py --dataset SMD --hidden_neurons 16 8 8 16 --batch_size 512 --epochs 100 --l2_regularizer 0.00001
python AutoEncoder_benchmark.py --dataset SWAT --hidden_neurons 16 8 8 16 --batch_size 512 --epochs 100 --l2_regularizer 0.1
python AutoEncoder_benchmark.py --dataset WADI --hidden_neurons 32 16 16 32 --batch_size 512 --epochs 100 --l2_regularizer 0.00001




python lstm_benchmark.py --dataset MSL --lr 0.001 --window_size 32 --stride 5 --num_layers 1 --hidden_size 256 --gpu 0
python lstm_benchmark.py --dataset SMAP --lr 0.001 --window_size 250 --stride 5 --num_layers 1 --hidden_size 32 --gpu 0
python lstm_benchmark.py --dataset SMD --lr 0.001 --window_size 32 --stride 5 --num_layers 1 --hidden_size 128 --gpu 0
python lstm_benchmark.py --dataset SWAT --lr 0.001 --window_size 32 --stride 5 --num_layers 1 --hidden_size 32 --gpu 0
python lstm_benchmark.py --dataset WADI --lr 0.001 --window_size 32 --stride 5 --num_layers 2 --hidden_size 256 --gpu 0
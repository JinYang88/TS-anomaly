python AutoEncoder_benchmark.py --dataset SMD --hidden_neurons 32 16 16 32 --batch_size 512 --epochs 100 --l2_regularizer 0.00001
python AutoEncoder_benchmark.py --dataset SMD --hidden_neurons 16 8 8 16  --batch_size 512 --epochs 100 --l2_regularizer 0.00001

python AutoEncoder_benchmark.py --dataset SMD --hidden_neurons 32 16 16 32 --batch_size 512 --epochs 100 --l2_regularizer 0.1
python AutoEncoder_benchmark.py --dataset SMD --hidden_neurons 16 8 8 16  --batch_size 512 --epochs 100 --l2_regularizer 0.1
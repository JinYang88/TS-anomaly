CUDA_VISIBLE_DEVICES=1 python mad_gan_benchmark.py --dataset SWAT --window_size 32 --stride 5 --lr 0.005 --num_epochs 50
CUDA_VISIBLE_DEVICES=1 python mad_gan_benchmark.py --dataset SWAT --window_size 32 --stride 5 --lr 0.005 --num_epochs 150

CUDA_VISIBLE_DEVICES=1 python mad_gan_benchmark.py --dataset SWAT --window_size 64 --stride 5 --lr 0.005 --num_epochs 50
CUDA_VISIBLE_DEVICES=1 python mad_gan_benchmark.py --dataset SWAT --window_size 64 --stride 5 --lr 0.005 --num_epochs 150


CUDA_VISIBLE_DEVICES=4 python mad_gan_benchmark.py --dataset SWAT --window_size 32 --stride 5 --lr 0.001 --num_epochs 50
CUDA_VISIBLE_DEVICES=4 python mad_gan_benchmark.py --dataset SWAT --window_size 32 --stride 5 --lr 0.001 --num_epochs 150

CUDA_VISIBLE_DEVICES=4 python mad_gan_benchmark.py --dataset SWAT --window_size 64 --stride 5 --lr 0.001 --num_epochs 50
CUDA_VISIBLE_DEVICES=4 python mad_gan_benchmark.py --dataset SWAT --window_size 64 --stride 5 --lr 0.001 --num_epochs 150
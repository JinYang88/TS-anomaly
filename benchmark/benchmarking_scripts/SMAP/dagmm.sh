# sh benchmarking_scripts/SMAP/dagmm.sh > logs/dagmm.SMAP.bench 2>&1 &

export PATH=$PATH:/research/dept7/jyliu/cuda/cuda-9.0/bin
export LD_LIBRARY_PATH=/research/dept7/jyliu/cuda/cuda-9.0/lib64

python dagmm_benchmark.py --dataset SMAP --lr 0.0001 --dropout 0.25 --num_epochs 100 -ch 32 16 2 -eh 80 40
python dagmm_benchmark.py --dataset SMAP --lr 0.0003 --dropout 0.25 --num_epochs 100 -ch 32 16 2 -eh 80 40


python dagmm_benchmark.py --dataset SMAP --lr 0.0001 --dropout 0.25 --num_epochs 100 -ch 64 32 2 -eh 60 30
python dagmm_benchmark.py --dataset SMAP --lr 0.0003 --dropout 0.25 --num_epochs 100 -ch 64 32 2 -eh 60 30
# 

python dagmm_benchmark.py --dataset SMAP --lr 0.0001 --dropout 0.25 --num_epochs 100 -ch 128 64 2 -eh 100 50
python dagmm_benchmark.py --dataset SMAP --lr 0.0003 --dropout 0.25 --num_epochs 100 -ch 128 64 2 -eh 100 50
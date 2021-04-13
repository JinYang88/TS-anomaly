# sh benchmarking_scripts/SMD/dagmm.sh > logs/dagmm.SMD.bench 2>&1 &

python dagmm_benchmark.py --dataset SMD --lr 0.0001 --dropout 0.25 --num_epochs 100 -ch 32 16 2 -eh 80 40
python dagmm_benchmark.py --dataset SMD --lr 0.0003 --dropout 0.25 --num_epochs 100 -ch 32 16 2 -eh 80 40


python dagmm_benchmark.py --dataset SMD --lr 0.0001 --dropout 0.25 --num_epochs 100 -ch 64 32 2 -eh 60 30
python dagmm_benchmark.py --dataset SMD --lr 0.0003 --dropout 0.25 --num_epochs 100 -ch 64 32 2 -eh 60 30
# 

python dagmm_benchmark.py --dataset SMD --lr 0.0001 --dropout 0.25 --num_epochs 100 -ch 128 64 2 -eh 100 50
python dagmm_benchmark.py --dataset SMD --lr 0.0003 --dropout 0.25 --num_epochs 100 -ch 128 64 2 -eh 100 50
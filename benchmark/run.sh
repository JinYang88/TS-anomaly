python cmanomaly_benchmark.py --stride 1 --info try_strides --gpu 0 B
python cmanomaly_benchmark.py --stride 3 --info try_strides --gpu 0 B
python cmanomaly_benchmark.py --stride 5 --info try_strides --gpu 0 B
python cmanomaly_benchmark.py --stride 10 --info try_strides --gpu 0 B


python cmanomaly_benchmark.py --stride 16 --info try_strides --gpu 1 B
python cmanomaly_benchmark.py --stride 32 --info try_strides --gpu 1 B
python cmanomaly_benchmark.py --stride 64 --info try_strides --gpu 1 B
python cmanomaly_benchmark.py --stride 100 --info try_strides --gpu 1 B


python cmanomaly_benchmark.py --stride 1 --strategy uniform --info try_uniform --gpu 0 B
python cmanomaly_benchmark.py --stride 3 --strategy uniform --info try_uniform --gpu 0 B
python cmanomaly_benchmark.py --stride 5 --strategy uniform --info try_uniform --gpu 0 B
python cmanomaly_benchmark.py --stride 10 --strategy uniform --info try_uniform --gpu 0 B


python cmanomaly_benchmark.py --stride 16 --strategy uniform --info try_uniform --gpu 1 B
python cmanomaly_benchmark.py --stride 32 --strategy uniform --info try_uniform --gpu 1 B
python cmanomaly_benchmark.py --stride 64 --strategy uniform --info try_uniform --gpu 1 B
python cmanomaly_benchmark.py --stride 100 --strategy uniform --info try_uniform --gpu 1 B
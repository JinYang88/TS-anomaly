python iforest_benchmark.py --dataset MSL --n_estimators 16
python iforest_benchmark.py --dataset SMAP --n_estimators 150
python iforest_benchmark.py --dataset SMD --n_estimators 16
python iforest_benchmark.py --dataset SWAT --n_estimators 32
python iforest_benchmark.py --dataset WADI --n_estimators 64
python KNN_benchmark.py --dataset MSL --n_neighbors 5
python KNN_benchmark.py --dataset SMAP --n_neighbors 20
python KNN_benchmark.py --dataset SMD --n_neighbors 30
python KNN_benchmark.py --dataset SWAT --n_neighbors 30
python KNN_benchmark.py --dataset WADI_SPLIT --n_neighbors 3
python LODA_benchmark.py --dataset MSL --n_bins 40
python LODA_benchmark.py --dataset SMAP --n_bins 30
python LODA_benchmark.py --dataset SMD --n_bins 20
python LODA_benchmark.py --dataset SWAT --n_bins 40
python LODA_benchmark.py --dataset WADI --n_bins 40
python LOF_benchmark.py --dataset MSL --n_neighbors 5
python LOF_benchmark.py --dataset SMAP --n_neighbors 10
python LOF_benchmark.py --dataset SMD --n_neighbors 30
python LOF_benchmark.py --dataset SWAT --n_neighbors 3
python LOF_benchmark.py --dataset WADI_SPLIT --n_neighbors 3
python PCA_benchmark.py --dataset MSL --n_components 25
# possible buggy
# python PCA_benchmark.py --dataset SMAP --n_components 10
python PCA_benchmark.py --dataset SMAP --n_components 5 B
python PCA_benchmark.py --dataset SMAP --n_components 10 B
python PCA_benchmark.py --dataset SMAP --n_components 15 B
python PCA_benchmark.py --dataset SMAP --n_components 25 B

python PCA_benchmark.py --dataset SMD --n_components 5
python PCA_benchmark.py --dataset SWAT --n_components 20
python PCA_benchmark.py --dataset WADI_SPLIT --n_components 10
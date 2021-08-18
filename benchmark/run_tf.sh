python 6_lstm_vae_benchmark.py --dataset SMD --lr 0.001 --z_dim 3 --intermediate_dim 64 --window_size 32 --stride 5 --hidden_size 128 --num_epochs 10
python 7_dagmm_benchmark.py --dataset SMD  -ch 32 16 2 -eh 80 40
python 8_omnianomaly_benchmark.py --dataset SMD --lr 0.001 --z_dim 3 --rnn_num_hidden 500 --window_size 32 --stride 5 --dense_dim 128 --nf_layers 2 --max_epoch 1


python 0_3sigma_benchmark.py --dataset SMD 
python 1_iforest_benchmark.py --dataset SMD --n_estimators 100
python 2_LODA_benchmark.py --dataset SMD --n_bins 10
python 3_PCA_benchmark.py --dataset SMD --n_components 10
python 4_AutoEncoder_benchmark.py --dataset SMD --hidden_neurons 64 32 32 64 --batch_size 32 --epochs 100 --l2_regularizer 0.1
python 5_lstm_benchmark.py --dataset SMD --lr 0.001 --window_size 32 --stride 5 --num_layers 2 --hidden_size 64 --gpu -1
python 9_cmanomaly_benchmark.py --dataset SMD --lr 0.001 --window_size 64 --stride 5 --embedding_dim 16 --nbins 10 --gpu 0
python 10_cmanomaly_old_benchmark.py --dataset SMD --lr 0.001 --window_size 64 --stride 5 --gpu 0

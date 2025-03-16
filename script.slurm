#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -p gpu
#SBATCH -C gmem24
#SBATCH --job-name=ntu60cv
#SBATCH --output=./logs/%x.out

# Load modules
module load anaconda3
module list                            # Have Nvidia tell us the GPU/CPU mapping so we know
module load cuda/11.7
nvidia-smi topo -m

python3 main.py --train_classifier --gpu 0 --run_id ntu60cv --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 4 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM

python3 main.py --test_classifier --gpu 0 --load_model ntu60cv_18-02-25_0136 --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 4 --num_workers 8

python3 main.py --train_classifier --gpu 0 --run_id lab639_test --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 4 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM
python3 main.py --train_classifier --gpu 0 --run_id fisheye639_CV --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 4 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM
python3 main.py --train_classifier --gpu 0 --run_id fisheye639_CV_train_1_2_test_3_4_test --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 4 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM
python3 main.py --train_classifier --gpu 0 --run_id fisheye639_CV_train_2_3_4_test_1_test --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 4 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM
python3 main.py --train_classifier --gpu 0 --run_id fisheye639_CV_train_1_3_4_test_2_test --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 4 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM

python3 main.py --train_classifier --gpu 0 --run_id lab639_test_fold1 --dataset ntu_rgbd_60_fold1 --model_version 'v3' --batch_size 4 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM

python3 main.py --test_classifier --gpu 0 --load_model lab639_test_04-03-25_0122 --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 4 --num_workers 8
python3 main.py --test_classifier --gpu 0 --load_model lab639_test_04-03-25_0258 --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 4 --num_workers 8

python3 main.py --test_classifier --gpu 0 --load_model ntu_rgbd_60_fold0_10-03-25_1657 --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 4 --num_workers 8
python3 main.py --test_classifier --gpu 0 --load_model ntu_rgbd_60_fold1_10-03-25_1917 --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 4 --num_workers 8
python3 main.py --test_classifier --gpu 0 --load_model ntu_rgbd_60_fold2_10-03-25_2136 --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 4 --num_workers 8
python3 main.py --test_classifier --gpu 0 --load_model ntu_rgbd_60_fold3_10-03-25_2356 --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 4 --num_workers 8
python3 main.py --test_classifier --gpu 0 --load_model ntu_rgbd_60_fold4_11-03-25_0216 --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 4 --num_workers 8
python3 main.py --test_classifier --gpu 0 --load_model fisheye639_DIF_11-03-25_1202 --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 4 --num_workers 8
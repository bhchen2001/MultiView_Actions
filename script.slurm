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

python3 main.py --train_classifier --gpu 0 --run_id ntu60cv --dataset ntu_rgbd_60 --model_version 'v3' --batch_size 8 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM

python3 main.py --train_classifier --gpu 0 --run_id numa_cs_test --dataset numa_cs --model_version 'v3' --batch_size 4 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM
python3 main.py --train_classifier --gpu 0 --run_id numa_cv_test --dataset numa_cv --model_version 'v3' --batch_size 4 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM
python3 main.py --train_classifier --gpu 0 --run_id numa_cv_remove_loss_ac_vc --dataset numa_cv --model_version 'v3' --batch_size 4 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM
python3 main.py --train_classifier --gpu 0 --run_id numa_cs_remove_loss_ac_vc --dataset numa_cs --model_version 'v3' --batch_size 4 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM
python3 main.py --train_classifier --gpu 0 --run_id numa_cs_remove_loss_vl_vc_vq --dataset numa_cs --model_version 'v3' --batch_size 4 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM
python3 main.py --train_classifier --gpu 0 --run_id numa_cv_remove_loss_vl_vc_vq --dataset numa_cv --model_version 'v3' --batch_size 4 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM

python3 main.py --train_classifier --gpu 0 --run_id numa_single_view1 --dataset numa_view --model_version 'v3' --batch_size 8 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM
python3 main.py --train_classifier --gpu 0 --run_id numa_single_view2 --dataset numa_view --model_version 'v3' --batch_size 8 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM
python3 main.py --train_classifier --gpu 0 --run_id numa_single_view3 --dataset numa_view --model_version 'v3' --batch_size 8 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM

python3 main.py --train_classifier --gpu 0 --run_id numa_all_view --dataset numa_view --model_version 'v3' --batch_size 4 --num_epochs 51 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM

python3 main.py --test_classifier --gpu 0 --load_model numa_cs_test_10-01-25_2153 --dataset numa_cs --model_version 'v3' --batch_size 4 --num_workers 8
python3 main.py --test_classifier --gpu 0 --load_model numa_cv_test_11-01-25_1315 --dataset numa_cv --model_version 'v3' --batch_size 4 --num_workers 8
python3 main.py --test_classifier --gpu 0 --load_model numa_cv_remove_loss_ac_vc_11-01-25_1621 --dataset numa_cv --model_version 'v3' --batch_size 4 --num_workers 8
python3 main.py --test_classifier --gpu 0 --load_model numa_cs_remove_loss_ac_vc_11-01-25_1753 --dataset numa_cs --model_version 'v3' --batch_size 4 --num_workers 8
python3 main.py --test_classifier --gpu 0 --load_model numa_cs_remove_loss_vl_vc_vq_12-01-25_1643 --dataset numa_cs --model_version 'v3' --batch_size 4 --num_workers 8 
python3 main.py --test_classifier --gpu 0 --load_model numa_cv_remove_loss_vl_vc_vq_12-01-25_2011 --dataset numa_cv --model_version 'v3' --batch_size 4 --num_workers 8

python3 main.py --test_classifier --gpu 0 --load_model numa_single_view1_17-01-25_1310 --dataset numa_view --model_version 'v3' --batch_size 4 --num_workers 8
python3 main.py --test_classifier --gpu 0 --load_model numa_single_view2_17-01-25_1334 --dataset numa_view --model_version 'v3' --batch_size 4 --num_workers 8
python3 main.py --test_classifier --gpu 0 --load_model numa_single_view3_17-01-25_1451 --dataset numa_view --model_version 'v3' --batch_size 4 --num_workers 8

python3 main.py --test_classifier --gpu 0 --load_model numa_all_view_17-01-25_1608 --dataset numa_view --model_version 'v3' --batch_size 4 --num_workers 8
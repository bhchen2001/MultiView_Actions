for fold in {1..4}; do
    # python3 main.py --train_classifier --gpu 0 --run_id fisheye639_S003_fold${fold} --dataset ntu_rgbd_60_fold${fold} --model_version 'v3' --batch_size 4 --num_epochs 30 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM
    python3 main.py --train_classifier --gpu 0 --run_id fisheye639_S003_CV${fold} --dataset ntu_rgbd_60_CV${fold} --model_version 'v3' --batch_size 4 --num_epochs 30 --num_workers 8 --learning_rate 1e-4 --weight_decay 1e-6 --optimizer ADAM
    # python3 main.py --test_classifier --gpu 0 --load_model fisheye639_S003_fold${fold} --dataset ntu_rgbd_60_fold${fold} --model_version 'v3' --batch_size 4 --num_workers 8
done
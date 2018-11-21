#!/usr/bin/env bash
model="pcnn"
devices="1"
lr="0.5"
mkdir ./checkpoint/$model/
# for lr in 0.5 0.1 0.01 0.001 0.0001
# do
    for dropout in 0.5 0.4 0.3 0.2 0.1
    do
        mkdir -p ./checkpoint/$model/lr$lr-dp$dropout
        mkdir -p ./summary/$model/lr$lr-dp$dropout
        echo "learning rate = $lr"
        echo "dropout = $dropout"
        CUDA_VISIBLE_DEVICES=$devices python train.py --model_name $model --learning_rate $lr --drop_prob $dropout --checkpoint_dir ./checkpoint/$model/lr$lr-dp$dropout --summary_dir ./summary/$model/lr$lr-dp$dropout > ./checkpoint/$model/lr$lr-dp$dropout/train_log.txt
        echo "start evaluating on test......"
        CUDA_VISIBLE_DEVICES=$devices python test.py --model_name $model --learning_rate $lr --drop_prob $dropout --checkpoint_dir ./checkpoint/$model/lr$lr-dp$dropout > ./checkpoint/$model/lr$lr-dp$dropout/test_log.txt
        echo "start evaluating on dev......"
        CUDA_VISIBLE_DEVICES=$devices python dev.py --model_name $model --learning_rate $lr --drop_prob $dropout --checkpoint_dir ./checkpoint/$model/lr$lr-dp$dropout > ./checkpoint/$model/lr$lr-dp$dropout/dev_log.txt
    done
# done

echo "tuning for $model finished!"

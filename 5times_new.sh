devices=$1
model=$2
dropout=$3
lr=$4
bsize=$5

for i in 1 2 3 4 5
do
	echo "Run #$i"
	mkdir ./checkpoint/$model/lr$lr-dp$dropout/r$i
	CUDA_VISIBLE_DEVICES=$devices python train.py --model_name $model --drop_prob $dropout --learning_rate $lr --batch_size $bsize --random_seed $i  --checkpoint_dir ./checkpoint/$model/lr$lr-dp$dropout/r$i > ./checkpoint/$model/lr$lr-dp$dropout/r$i/train_log.txt --max_epoch 100
	echo "start evaluating on dev......"
	CUDA_VISIBLE_DEVICES=$devices python dev.py --model_name $model --drop_prob $dropout --learning_rate $lr --batch_size $bsize --random_seed $i --checkpoint_dir ./checkpoint/$model/lr$lr-dp$dropout/r$i > ./checkpoint/$model/lr$lr-dp$dropout/r$i/dev_log.txt --epoch_range "(5,100)"
	echo "start evaluating on test......"
	CUDA_VISIBLE_DEVICES=$devices python test.py --model_name $model --drop_prob $dropout --learning_rate $lr --batch_size $bsize --random_seed $i --checkpoint_dir ./checkpoint/$model/lr$lr-dp$dropout/r$i > ./checkpoint/$model/lr$lr-dp$dropout/r$i/test_log.txt --epoch_range "(5,100)"
#	python modify_name.py $model $i $lr $dropout $bsize
done

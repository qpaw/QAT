#!/bin/bash

val_dataset="/home/user/sQAT/datasets/ImageNet/val/"

target_dir="/home/user/sQAT/datasets/ImageNet/val_small/"
$(mkdir $target_dir)

list_of_dirs=$(ls $val_dataset)
# echo "$list_of_dirs"

for val in $list_of_dirs
do
	list_of_images=$(ls $val_dataset$val)
	$(mkdir "$target_dir$val")
	for im in $list_of_images
	do
		#echo "$val_dataset$val ---- $im"
		$(cp "$val_dataset$val/$im" "$target_dir$val" )
		break
	done
done

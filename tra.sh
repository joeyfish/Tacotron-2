#!/bin/bash

basepath=$(cd `dirname $0`; pwd)
echo $basepath
cd $basepath

#mkdir -p $basepath/logs
#CUDA_VISIBLE_DEVICES=1 python3 -m pdb $basepath/train.py 
#CUDA_VISIBLE_DEVICES=1 python3 $basepath/train.py --restore_step 2000 --model Tacotron-2
#CUDA_VISIBLE_DEVICES=1 python3 $basepath/train.py --restore_step 2000 --model WaveNet
python3 $basepath/train.py --tacotron_train_steps 220000

#CUDA_VISIBLE_DEVICES=1 python3 -u $basepath/train.py > $basepath/logs/daemon.log 2>&1 & 

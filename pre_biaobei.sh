#!/bin/bash
basepath=$(cd `dirname $0`; pwd)
echo $basepath
cd $basepath

rm ../backup_models -rf
mkdir -p ../backup_models
cp logs-Tacotron-2 ../backup_models -R
rm training_data logs-Tacotron logs-Tacotron-2 -rf
echo "rm training_data logs-Tacotron logs-Tacotron-2 -rf"
#all
#python3 ./preprocess.py --dataset corpus --subdataset THCHS-30 --speaker_thchs30 a
#female
#python3 ./preprocess.py --dataset corpus --subdataset THCHS-30 --speaker_thchs30 f
#male
#python3 ./preprocess.py --dataset corpus --subdataset THCHS-30 --speaker_thchs30 m
python3 ./preprocess.py --dataset corpus --subdataset BIAOBEI
# female max 214, male max 186 "a" for all, eg. "m12" for num 12 male, "f11" for num 11 female
#rm -f ~/temp/aishelldebug* -v
#python3 ./preprocess.py --dataset corpus --subdataset AISHELL --speaker_aishell m5
#python3 ./preprocess.py --dataset corpus --subdataset AISHELL --speaker_aishell f12
#python3 ./preprocess.py --dataset corpus --subdataset mycorpus

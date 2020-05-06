#!/bin/bash

mkdir -p ~/temp/backup
i=0
N=$((10/1))
for file in `find $1 -name "*.wav"`
do
i=`expr $i + 1`
mv $file $file.bak.wav
python /home/user/mycode/transwav.16000.py $file.bak.wav $file
mv $file.bak.wav ~/temp/backup
[ $(($i%$N)) -eq 0 ] && echo -ne $i"-"
done

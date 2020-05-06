#echo "find result at folder Tacotron-2-Chinese/tacotron_output/logs-eval/wavs"
rm tacotron_output/logs-eval/* -rf
#python3 synthesize.py --model='Tacotron' --text='sentences_cn.txt' --eval_seperate
#python3 synthesize.py --model='Tacotron' --text='sentences_sh.txt' --eval_seperate
python3 synthesize.py --model='Tacotron' --text='sentences_sh.txt'
echo "find result at folder ./tacotron_output/logs-eval/"
echo "cp tacotron_output/logs-eval/wavs/* ../evals/051000"


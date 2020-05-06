basepath=$(cd `dirname $0`; pwd)
echo $basepath
ls $basepath/corpus
./download_and_untar.sh $basepath/corpus www.openslr.org/resources/33 data_aishell
./download_and_untar.sh $basepath/corpus www.openslr.org/resources/33 resource_aishell

#if [ -d $basepath/corpus/data_aishell/wave ] 
if [ -f $basepath/corpus/data_aishell/transcript/000001-010000.txt ] 
then
    echo "$0: data part $part was already successfully extracted, nothing to do."
else

    echo "$0: data part $part was not here."

    rm $basepath/corpus/data_aishell/wave/ -rf        
    
    mkdir -p $basepath/corpus/data_aishell/wave/
    cd $basepath/corpus/data_aishell/wav/
    echo "please wait for minutes........"
    find -name "*.wav" | xargs -i mv {} $basepath/corpus/data_aishell/wave/        
    #find -name "*.wav" | xargs -i cp {} $basepath/corpus/data_aishell/wave/        
    cd $basepath
    python ./data_aishell.py --src $basepath/corpus/data_aishell/transcript/aishell_transcript_v0.8.txt \
            --dst $basepath/corpus/data_aishell/transcript/000001-010000.txt
fi
import os, sys
from argparse import ArgumentParser
from pinyin2cn import *

def data_aishell(args):
    if args.source_name != False and args.out_name != False :
        jieba.load_userdict("user_dict/jieba1.txt")
        jieba.load_userdict("user_dict/jieba.txt")

        file1 = os.path.normpath(args.source_name)
        assert os.path.exists(file1), 'Source file not exist!'
        #file2 = os.path.normpath(args.out_name)
        #assert os.path.isfile(file2), 'out file does not exist!'
        f1 = open(file1, "r")
        txts = f1.read().split("\n")
        f1.close()
        f2 = open(args.out_name, "w", encoding = "utf8")
        ind = 1
        total = len(txts)
        for tx in txts:
            if tx == "":
                continue
            index1 = tx.index(" ")
            fileinfo = tx[:index1]
            out1 = tx[index1:]
            content1 = cn2pinyin2(out1)    
            content2 = cn2pinyin(out1)    
            #print (out1, content)
            f2.write(f'{fileinfo}\t{content1}\n')
            f2.write(f'\t{content2}\n')
            ind += 1
            if (ind % 2000) == 0:
                print(f'{ind:06d} / {total:06d} : {out1} ')
        f2.close()    

        
        
def main():
    parser = ArgumentParser()
    parser.add_argument('-s', '--src', dest='source_name', default = False, help='src aishell_transcript_v0.8.txt', required=True)
    parser.add_argument('-d', '--dst', dest='out_name', default = False, help='dst 000001-010000.txt', required=True)
    args = parser.parse_args()
    data_aishell(args)

if __name__ == '__main__':
	main()



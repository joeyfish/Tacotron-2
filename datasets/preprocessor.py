import glob, os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from datasets import audio
import sys, time

MAX_WAV_AISHELL = 1500
def build_from_path_thchs30(args, hparams, input_dirs, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited

	#total 60 people, A 19 people, B 14 people, C 17 people, D 10 people
	#http://166.111.134.19:7777/data/thchs30/README.html
	#A11,A12,A13,A14,A19,A2,A22,A23,A32,A33,A34,A35,A36,A4,A5,A6,A7,A8,A9,
	#B11,B12,B15,B2,B21,B22,B31,B32,B33,B34,B4,B6,B7,B8,
	#C12,C13,C14,C17,C18,C19,C2,C20,C21,C22,C23,C31,C32,C4,C6,C7,C8,D11,D12,D13,D21,D31,D32,D4,D6,D7,D8
	# total 13388 wav files
	'''
	  kaldi script
      for nn in `find  $corpus_dir/$x -name "*.wav" | sort -u | xargs -I {} basename {} .wav`; do
          spkid=`echo $nn | awk -F"_" '{print "" $1}'`
          spk_char=`echo $spkid | sed 's/\([A-Z]\).*/\1/'`
          spk_num=`echo $spkid | sed 's/[A-Z]\([0-9]\)/\1/'`
          spkid=$(printf '%s%.2d' "$spk_char" "$spk_num")
          utt_num=`echo $nn | awk -F"_" '{print $2}'`
          uttid=$(printf '%s%.2d_%.3d' "$spk_char" "$spk_num" "$utt_num")
          echo $uttid $corpus_dir/$x/$nn.wav >> wav.scp
          echo $uttid $spkid >> utt2spk
          echo $uttid `sed -n 1p $corpus_dir/data/$nn.wav.trn` >> word.txt
          echo $uttid `sed -n 3p $corpus_dir/data/$nn.wav.trn` >> phone.txt
      done
      counts of every speaker id
        A11 = 250 female
        A12 = 150 female
        A13 = 250 female
        A14 = 250 female
        A19 = 250 female
        A2 = 249 female
        A22 = 250 female
        A23 = 250 female
        A32 = 250 female
        A33 = 110 male
        A34 = 249 female
        A35 = 7 male
        A36 = 250 female
        A4 = 250 female
        A5 = 5 male
        A6 = 250 female
        A7 = 249 female
        A8 = 250 male
        A9 = 13 male
        B11 = 250 female
        B12 = 250 female
        B15 = 250 female
        B2 = 249 female
        B21 = 19 male
        B22 = 250 female
        B31 = 250 female
        B32 = 250 female
        B33 = 250 female
        B34 = 7 male
        B4 = 250 female
        B6 = 248 female
        B7 = 249 female
        B8 = 250 male
        C12 = 250 female
        C13 = 250 female
        C14 = 250 female
        C17 = 250 female
        C18 = 250 female
        C19 = 250 female
        C2 = 90 female
        C20 = 250 female
        C21 = 250 female
        C22 = 250 female
        C23 = 250 female
        C31 = 250 female
        C32 = 250 female
        C4 = 250 female
        C6 = 250 female
        C7 = 249 female
        C8 = 250 male
        D11 = 249 female
        D12 = 248 female
        D13 = 250 female
        D21 = 250 female
        D31 = 250 female
        D32 = 249 female
        D4 = 250 female
        D6 = 250 female
        D7 = 249 female
        D8 = 250 male
        
        all 13388  male 1161  female 12227
        
        count of A B C D group
        A 3782  09:38:31 #train dev 
        B 3022  07:51:18 #train dev 
        C 4089  10:21:01 #train dev 
        D 2495  06:18:41 #test
	'''

	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	
	#index = 1

	for input_dir in input_dirs:
		trn_files_male_lst = []
		trn_files_male_a33 = glob.glob(os.path.join(input_dir, 'data', "A33*.trn"))
		trn_files_male_lst += trn_files_male_a33
		trn_files_male_a35 = glob.glob(os.path.join(input_dir, 'data', "A35*.trn"))
		trn_files_male_lst += trn_files_male_a35
		trn_files_male_a5 = glob.glob(os.path.join(input_dir, 'data', "A5*.trn"))
		trn_files_male_lst += trn_files_male_a5
		trn_files_male_a8 = glob.glob(os.path.join(input_dir, 'data', "A8*.trn"))
		trn_files_male_lst += trn_files_male_a8
		trn_files_male_a9 = glob.glob(os.path.join(input_dir, 'data', "A9*.trn"))
		trn_files_male_lst += trn_files_male_a9
		trn_files_male_b21 = glob.glob(os.path.join(input_dir, 'data', "B21*.trn"))
		trn_files_male_lst += trn_files_male_b21
		trn_files_male_b34 = glob.glob(os.path.join(input_dir, 'data', "B34*.trn"))
		trn_files_male_lst += trn_files_male_b34
		trn_files_male_b8 = glob.glob(os.path.join(input_dir, 'data', "B8*.trn"))
		trn_files_male_lst += trn_files_male_b8
		trn_files_male_c8 = glob.glob(os.path.join(input_dir, 'data', "C8*.trn"))
		trn_files_male_lst += trn_files_male_c8
		trn_files_male_d8 = glob.glob(os.path.join(input_dir, 'data', "D8*.trn"))
		trn_files_male_lst += trn_files_male_d8

		trn_files_all_lst = glob.glob(os.path.join(input_dir, 'data', "*.trn"))
		trn_files_female_lst = []
		for f in trn_files_all_lst:
			if f not in trn_files_male_lst:
				trn_files_female_lst.append(f)
		#print(f"all {len(trn_files_all_lst)}  male {len(trn_files_male_lst)}  female {len(trn_files_female_lst)}")

		trn_files = []
		if args.speaker_thchs30 == "a": #all
			trn_files = trn_files_all_lst
		elif args.speaker_thchs30 == "m": #male
			trn_files = trn_files_male_lst
		elif args.speaker_thchs30 == "f": #female
			trn_files = trn_files_female_lst
	
		for trn in trn_files:
			with open(trn) as f:
				text = f.readline().strip('\n')
				wav_path = trn[:-4]  # trn filename: A11_001.wav.trn
				basename = os.path.basename(wav_path)
				futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams)))
				index = index + 1
	return [future.result() for future in tqdm(futures) if future.result() is not None]

# biaobei & mycorpus
def build_from_path(args, hparams, input_dirs, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	for input_dir in input_dirs:
		with open(os.path.join(input_dir, 'ProsodyLabeling', '000001-010000.txt'), encoding='utf-8') as f:
			lines = f.readlines()
			index = 1

			sentence_index = ''
			sentence_pinyin = ''

			for line in lines:
				if line[0].isdigit():
					sentence_index = line[:6]
				else:
					sentence_pinyin = line.strip()
					wav_path = os.path.join(input_dir, 'Wave', '%s.wav' % sentence_index)
					futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, sentence_index, wav_path, sentence_pinyin, hparams)))
					index = index + 1
	return [future.result() for future in tqdm(futures) if future.result() is not None]

#female from 1 to 214, male from 1 to 186 "a" for all, \n
#eg. "m12" start from num 12 male, "f11" start from num 11 female, min wav 5000 files
#0176 F
#0177 F
def getwav_aishell_list(args, input_dir):
	g_running_path = os.path.abspath(os.path.dirname(sys.argv[0]))
	speaker_info_file = os.path.join(g_running_path, "user_dict", "speaker.aishell.info")
	fread = open(speaker_info_file, "r")
	readtxts_speaker_info = fread.read().split("\n")
	fread.close()

	speakers1 = args.speaker_aishell
	peopleindex = int(speakers1[1:])
	wav_files_list = []    
	if speakers1 == 'a':
		wav_files_list = glob.glob(os.path.join(input_dir, 'wave', "*.wav"))
	elif speakers1[0] == 'm':
		ind = 0
		for txt in readtxts_speaker_info:
			if txt.find('M') >= 0:
				ind += 1
				if ind >= peopleindex:
					search = txt[:4]
					wav_files_list += glob.glob(os.path.join(input_dir, 'wave', f"BAC009S{search}*.wav"))
					if len(wav_files_list) > MAX_WAV_AISHELL:
					    break;
	elif speakers1[0] == 'f':
		ind = 0
		for txt in readtxts_speaker_info:
			if txt.find('F') >= 0:
				ind += 1
				if ind >= peopleindex:
					search = txt[:4]
					wav_files_list += glob.glob(os.path.join(input_dir, 'wave', f"BAC009S{search}*.wav"))                    
					if len(wav_files_list) > MAX_WAV_AISHELL:
					    break;
	assert (len(wav_files_list) > 0)
	print(f'[{sys._getframe().f_lineno}] length of wav_files_list {len(wav_files_list)}')
	return wav_files_list
    
def getwav_path_aishell(wav_list, sentence_index):
    #print(f'[{sys._getframe().f_lineno}] wav_list {wav_list} sentence_index {sentence_index}')
    for f in wav_list:
        if f.find(sentence_index) >= 0:
            return f
    assert(False)
    return ""        
    
def build_from_path_aishell(args, hparams, input_dirs, mel_dir, linear_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	start=time.time()
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	for input_dir in input_dirs:
		with open(os.path.join(input_dir, 'transcript', '000001-010000.txt'), encoding='utf-8') as f:
			lines = f.readlines()
			#index = 1

			wav_list = getwav_aishell_list(args, input_dir)

			sentence_index = ''
			sentence_pinyin = ''

			for line in lines:
				if line[0] == 'B':
					sentence_index = os.path.join(input_dir, 'wave', line[:16] + '.wav')
				else:
					if sentence_index not in wav_list:
						continue
					wav_path = ""
					sentence_pinyin = line.strip()
					#print(f'sentence_index = {sentence_index} input_dir= {input_dir}')
					wav_path = getwav_path_aishell(wav_list, sentence_index)
					basename = wav_path[:-4]
					wav_file = basename + '.wav'
					wav_path = wav_file
					basename = basename.split('/')[-1]
					text = sentence_pinyin
					#futures.append(executor.submit(partial(_process_utterance, wav_dir, mel_dir, basename, wav_path, text, hparams)))
					futures.append(executor.submit(partial(_process_utterance, mel_dir, linear_dir, wav_dir, basename, wav_path, text, hparams)))
					index += 1
					if (index % 1000) == 0:
					    print(f'[{sys._getframe().f_lineno}] index {index}/{len(wav_list)} file {wav_path} text {text}')
					if index > MAX_WAV_AISHELL:
					    break    
	end=time.time()-start
	print('time cost %0.2f'%end)
	return [future.result() for future in tqdm(futures) if future.result() is not None]

def _process_utterance(mel_dir, linear_dir, wav_dir, index, wav_path, text, hparams):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
	"""
	try:
		# Load the audio as numpy array
		wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
	except FileNotFoundError: #catch missing wav exception
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
			wav_path))
		return None

	#rescale wav
	if hparams.rescale:
		wav = wav / np.abs(wav).max() * hparams.rescaling_max

	#M-AILABS extra silence specific
	if hparams.trim_silence:
		wav = audio.trim_silence(wav, hparams)

	#[-1, 1]
	out = wav
	constant_values = 0.
	out_dtype = np.float32

	# Compute the mel scale spectrogram from the wav
	mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
	mel_frames = mel_spectrogram.shape[1]

	if mel_frames > hparams.max_mel_frames or len(text) > hparams.max_text_length:
		return None

	#Compute the linear scale spectrogram from the wav
	linear_spectrogram = audio.linearspectrogram(wav, hparams).astype(np.float32)
	linear_frames = linear_spectrogram.shape[1]

	#sanity check
	assert linear_frames == mel_frames

	#Ensure time resolution adjustement between audio and mel-spectrogram
	fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
	l, r = audio.pad_lr(wav, fft_size, audio.get_hop_size(hparams))

	#Zero pad for quantized signal
	out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
	assert len(out) >= mel_frames * audio.get_hop_size(hparams)

	#time resolution adjustement
	#ensure length of raw audio is multiple of hop size so that we can use
	#transposed convolution to upsample
	out = out[:mel_frames * audio.get_hop_size(hparams)]
	assert len(out) % audio.get_hop_size(hparams) == 0
	time_steps = len(out)

	# Write the spectrogram and audio to disk
	audio_filename = 'audio-{}.npy'.format(index)
	mel_filename = 'mel-{}.npy'.format(index)
	linear_filename = 'linear-{}.npy'.format(index)
	# np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
	np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)
	np.save(os.path.join(linear_dir, linear_filename), linear_spectrogram.T, allow_pickle=False)

	# Return a tuple describing this training example
	return (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, text)

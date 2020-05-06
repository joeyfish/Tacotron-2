import argparse
import os
from multiprocessing import cpu_count

from datasets import preprocessor
from hparams import hparams
from tqdm import tqdm


def preprocess(args, input_folders, out_dir, hparams):
	mel_dir = os.path.join(out_dir, 'mels')
	wav_dir = os.path.join(out_dir, 'audio')
	linear_dir = os.path.join(out_dir, 'linear')
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(linear_dir, exist_ok=True)
	if args.subdataset.startswith('THCHS-30'):
		metadata = preprocessor.build_from_path_thchs30(args, hparams, input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	elif args.subdataset.startswith('mycorpus') or args.subdataset.startswith('BIAOBEI'):
		metadata = preprocessor.build_from_path(args, hparams, input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	elif args.subdataset.startswith('AISHELL'):
		metadata = preprocessor.build_from_path_aishell(args, hparams, input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	else:
		metadata = preprocessor.build_from_path(args, hparams, input_folders, mel_dir, linear_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	assert len(metadata) > 0
	write_metadata(metadata, out_dir)

def write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	mel_frames = sum([int(m[4]) for m in metadata])
	timesteps = sum([int(m[3]) for m in metadata])
	sr = hparams.sample_rate
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
		len(metadata), mel_frames, timesteps, hours))
	print('Max input length (text chars): {}'.format(max(len(m[5]) for m in metadata)))
	print('Max mel frames length: {}'.format(max(int(m[4]) for m in metadata)))
	print('Max audio timesteps length: {}'.format(max(m[3] for m in metadata)))

def norm_data(args):

	merge_books = (args.merge_books=='True')

	print('Selecting data folders..')
	supported_datasets = ['LJSpeech-1.0', 'LJSpeech-1.1', 'M-AILABS', 'MANDARIN', 'corpus']
	if args.dataset not in supported_datasets:
		raise ValueError(f'dataset value entered {args.dataset} does not belong to supported datasets: {supported_datasets}')

	if args.dataset.startswith('LJSpeech'):
		return [os.path.join(args.base_dir, args.dataset)]

	if args.dataset.startswith('MANDARIN'):
		return [os.path.join(args.base_dir, 'data_mandarin', anchor) for anchor in hparams.anchor_dirs]

	if args.dataset.startswith('corpus'):
		supported_subdatasets = ['mycorpus', 'THCHS-30', 'BIAOBEI', 'AISHELL']
		if args.subdataset not in supported_subdatasets:
			raise ValueError(f' --subdataset {args.subdataset} does not belong to supported datasets: {supported_subdatasets}')
		if args.subdataset.startswith('THCHS-30'):
			#total 60 peaple, A 19 peaple, B 14 peaple, C 17 peaple, D 10 peaple
			#http://166.111.134.19:7777/data/thchs30/README.html
			#A11,A12,A13,A14,A19,A2,A22,A23,A32,A33,A34,A35,A36,A4,A5,A6,A7,A8,A9,
			#B11,B12,B15,B2,B21,B22,B31,B32,B33,B34,B4,B6,B7,B8,
			#C12,C13,C14,C17,C18,C19,C2,C20,C21,C22,C23,C31,C32,C4,C6,C7,C8,D11,D12,D13,D21,D31,D32,D4,D6,D7,D8
			if args.speaker_thchs30 == 'dummy':
			    raise ValueError(f'missing parameter: --speaker_thchs30 "a" for all, "m" for male, "f" for female')
			if args.speaker_thchs30 not in ["a","f","m"]:
				raise ValueError(f' --subdataset {args.speaker_thchs30} does not belong to supported datasets: ["a","f","m"]')
			return [os.path.join(args.base_dir, 'corpus', 'data_thchs30')]
		elif args.subdataset.startswith('BIAOBEI'):
			return [os.path.join(args.base_dir, 'corpus', 'BZNSYP')]
		elif args.subdataset.startswith('AISHELL'):
			if args.speaker_aishell == 'dummy':
			    raise ValueError(f'missing parameter: --speaker_aishell female from 1 to 214, male from 1 to 186 "a" for all, \neg. "m12" start from num 12 male, "f11" start from num 11 female, min wav 5000 files')
			return [os.path.join(args.base_dir, 'corpus', 'data_aishell')]
		elif args.subdataset.startswith('mycorpus'):
			return [os.path.join(args.base_dir, 'corpus', 'mycorpus')]

	if args.dataset == 'M-AILABS':
		supported_languages = ['en_US', 'en_UK', 'fr_FR', 'it_IT', 'de_DE', 'es_ES', 'ru_RU',
			'uk_UK', 'pl_PL', 'nl_NL', 'pt_PT', 'fi_FI', 'se_SE', 'tr_TR', 'ar_SA']
		if args.language not in supported_languages:
			raise ValueError(f'Please enter a supported language to use from M-AILABS dataset! \n{supported_languages}')

		supported_voices = ['female', 'male', 'mix']
		if args.voice not in supported_voices:
			raise ValueError(f'Please enter a supported voice option to use from M-AILABS dataset! \n{supported_voices}')

		path = os.path.join(args.base_dir, args.language, 'by_book', args.voice)
		supported_readers = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path,e))]
		if args.reader not in supported_readers:
			raise ValueError(f'Please enter a valid reader for your language and voice settings! \n{supported_readers}')

		path = os.path.join(path, args.reader)
		supported_books = [e for e in os.listdir(path) if os.path.isdir(os.path.join(path,e))]
		if merge_books:
			return [os.path.join(path, book) for book in supported_books]

		else:
			if args.book not in supported_books:
				raise ValueError(f'Please enter a valid book for your reader settings! \n{supported_books}')
			return [os.path.join(path, args.book)]


def run_preprocess(args, hparams):
	input_folders = norm_data(args)
	output_folder = os.path.join(args.base_dir, args.output)
	preprocess(args, input_folders, output_folder, hparams)


def main():
	print('initializing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--hparams', default='', help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	#parser.add_argument('--dataset', default='MANDARIN')
	parser.add_argument('--dataset', default='mycorpus')
	parser.add_argument('--subdataset', default='dummy', help='sub set have mycorpus, THCHS-30, BIAOBEI, AISHELL')
	parser.add_argument('--speaker_thchs30', default='dummy', help='"a" for all, "m" for male, "f" for female')
	parser.add_argument('--speaker_aishell', default='dummy', help='female from 1 to 214, male from 1 to 186 "a" for all, \neg. "m12" start from num 12 male, "f11" start from num 11 female, min wav 5000 files')
	parser.add_argument('--language', default='en_US')
	parser.add_argument('--voice', default='female')
	parser.add_argument('--reader', default='mary_ann')
	parser.add_argument('--merge_books', default='False')
	parser.add_argument('--book', default='northandsouth')
	parser.add_argument('--output', default='training_data')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	args = parser.parse_args()

	modified_hp = hparams.parse(args.hparams)

	assert args.merge_books in ('False', 'True')

	run_preprocess(args, modified_hp)


if __name__ == '__main__':
	main()

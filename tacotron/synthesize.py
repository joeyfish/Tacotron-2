import argparse
import os
import re
import time
import numpy as np
from time import sleep
from datasets import audio
import tensorflow as tf
from hparams import hparams, hparams_debug_string
from infolog import log
from tacotron.synthesizer import Synthesizer
from tqdm import tqdm


def generate_fast(model, text):
	model.synthesize(text, None, None, None, None)


def run_live(args, checkpoint_path, hparams):
	#Log to Terminal without keeping any records in files
	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams)
	synth.session_open()

	#Generate fast greeting message
	greetings = 'Hello, Welcome to the Live testing tool. Please type a message and I will try to read it!'
	log(greetings)
	generate_fast(synth, greetings)

	#Interaction loop
	while True:
		try:
			text = input()
			generate_fast(synth, text)

		except KeyboardInterrupt:
			leave = 'Thank you for testing our features. see you soon.'
			log(leave)
			generate_fast(synth, leave)
			sleep(2)
			break
	synth.session_close()


def run_eval(args, checkpoint_path, output_dir, hparams, sentences):
	eval_dir = os.path.join(output_dir, 'eval')
	log_dir = os.path.join(output_dir, 'logs-eval')

	if args.model == 'Tacotron-2':
		assert os.path.normpath(eval_dir) == os.path.normpath(args.mels_dir) #mels_dir = wavenet_input_dir

	#Create output path if it doesn't exist
	os.makedirs(eval_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'wavs'), exist_ok=True)
	os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)

	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams)
	synth.session_open()

	sentences = list(map(lambda s: s.strip(), sentences))
	delta_size = hparams.tacotron_synthesis_batch_size if hparams.tacotron_synthesis_batch_size < len(sentences) else len(sentences)
	batch_sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), delta_size)]
	start = time.time()
	for i, batch in enumerate(tqdm(batch_sentences)):
		mel_filename = os.path.join(eval_dir, f'{i:03d}.npy')
		mel = synth.eval(batch, args.speaker_id)
		np.save(mel_filename, mel.T, allow_pickle=False)
		wav = audio.inv_mel_spectrogram(mel.T, hparams)
		audio.save_wav(wav, os.path.join(eval_dir, f'{i:03d}.wav'), hparams)
	end = time.time() - start
	log(f'Generated total batch of {delta_size} in {end:.3f} sec')
	synth.session_close()


def run_synthesis(args, checkpoint_path, output_dir, hparams):
	GTA = (args.GTA == 'True')
	if GTA:
		synth_dir = os.path.join(output_dir, 'gta')

		#Create output path if it doesn't exist
		os.makedirs(synth_dir, exist_ok=True)
	else:
		synth_dir = os.path.join(output_dir, 'natural')

		#Create output path if it doesn't exist
		os.makedirs(synth_dir, exist_ok=True)

	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams, gta=GTA)
	synth.session_open()

	speaker_num = len(hparams.anchor_dirs)
	frame_shift_ms = hparams.hop_size / hparams.sample_rate
	for speaker_id in range(speaker_num):
		metadata_filename = os.path.join(args.input_dir, hparams.anchor_dirs[speaker_id], 'train.txt')
		with open(metadata_filename, encoding='utf-8') as f:
			metadata = [line.strip().split('|') for line in f]
		hours = sum([int(x[2]) for x in metadata]) * frame_shift_ms / 3600
		log(f'Loaded {hparams.anchor_dirs[speaker_id]} for {len(metadata)} examples ({hours:.2f} hours)')

		metadata = [metadata[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(metadata), hparams.tacotron_synthesis_batch_size)]
		if hparams.vocoder != 'melgan':
			mel_dir = os.path.join(args.input_dir, hparams.anchor_dirs[speaker_id], 'mels')
		else:
			mel_dir = os.path.join(args.input_dir, hparams.anchor_dirs[speaker_id])
		for meta in tqdm(metadata):
			texts = [m[3] for m in meta]
			mel_filenames = [os.path.join(mel_dir, m[0]) for m in meta]
			basenames = [os.path.basename(m).replace('.npy', '').replace('mel-', '') for m in mel_filenames]
			synth.synthesize(texts, basenames, synth_dir, None, mel_filenames, speaker_id)
	log(f'synthesized mel spectrograms at {synth_dir}')
	synth.session_close()


def tacotron_synthesize(args, hparams, checkpoint, sentences=None):
	output_dir = 'tacotron_' + args.output_dir

	try:
		checkpoint_path = tf.train.get_checkpoint_state(checkpoint).model_checkpoint_path
		log(f'loaded model at {checkpoint_path}')
	except:
		raise RuntimeError(f'Failed to load checkpoint at {checkpoint}')

	if args.mode == 'eval':
		run_eval(args, checkpoint_path, output_dir, hparams, sentences)
	elif args.mode == 'synthesis':
		run_synthesis(args, checkpoint_path, output_dir, hparams)
	else:
		run_live(args, checkpoint_path, hparams)

import os
import threading
import time
import traceback
import numpy as np
import tensorflow as tf
from infolog import log
from sklearn.model_selection import train_test_split
from tacotron.utils.text import text_to_sequence

_batches_per_group = 32 #must be integer multiplies of speaker_num

class Feeder:
	"""
	Feeds batches of data into queue on a background thread.
	"""
	def __init__(self, coordinator, metadata_dir, hparams):
		super(Feeder, self).__init__()
		self._coord = coordinator
		self._hparams = hparams
		self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]

		#Train test split
		if hparams.tacotron_test_size is None:
			assert hparams.tacotron_test_batches is not None

		test_size = (hparams.tacotron_test_size if hparams.tacotron_test_size is not None
			else hparams.tacotron_test_batches * hparams.tacotron_batch_size)

		#pad input sequences with the <pad_token> 0 ( _ )
		self._pad = 0
		#explicitely setting the padding to a value that doesn't originally exist in the spectogram
		#to avoid any possible conflicts, without affecting the output range of the model too much
		self._target_pad = -(hparams.max_abs_value + 1) if hparams.symmetric_mels else -1
		#Mark finished sequences with 1s
		self._token_pad = 1.

		self.speaker_num = len(self._hparams.anchor_dirs)
		self._train_offset = np.zeros(self.speaker_num, dtype=np.int32)
		self._test_offset = np.zeros(self.speaker_num, dtype=np.int32)
		self._metadata = []
		self._mel_dirs = [os.path.join(metadata_dir, anchor_dir, 'mels') for i, anchor_dir in enumerate(hparams.anchor_dirs)]

		# Load metadata
		frame_shift_ms = hparams.hop_size / hparams.sample_rate
		for i, anchor_dir in enumerate(hparams.anchor_dirs):
			with open(os.path.join(metadata_dir, anchor_dir, 'train.txt'), encoding='utf-8') as f:
				metadata = [line.strip().split('|') for line in f]
				self._metadata.append(metadata)
				hours = sum([int(x[2]) for x in metadata]) * frame_shift_ms / 3600
				log(f'Loaded {anchor_dir} for {len(metadata)} examples ({hours:.2f} hours)')

		# training and test set indices
		indices = [np.arange(len(self._metadata[i])) for i in range(self.speaker_num)]
		self.train_indices = []
		self.test_indices = []
		for i in range(self.speaker_num):
			train_indice, test_indice = train_test_split(indices[i], test_size=test_size, random_state=hparams.tacotron_data_random_state)
			self.train_indices.append(train_indice)
			self.test_indices.append(test_indice)

		#Make sure test_indices is a multiple of batch_size else round up
		len_test_indices = [self._round_down(len(self.test_indices[i]), hparams.tacotron_batch_size) for i in range(self.speaker_num)]
		self.extra_test = [self.test_indices[i][len_test_indices[i]:] for i in range(self.speaker_num)]
		for i in range(self.speaker_num):
			self.test_indices[i] = self.test_indices[i][:len_test_indices[i]]
			self.train_indices[i] = np.concatenate([self.train_indices[i], self.extra_test[i]])

		# training and test set split
		self._train_meta = [list(np.array(self._metadata[i])[self.train_indices[i]]) for i in range(self.speaker_num)]
		self._test_meta = [list(np.array(self._metadata[i])[self.test_indices[i]])  for i in range(self.speaker_num)]
		self.test_steps = sum(len(self._test_meta[i]) for i in range(len(self._test_meta))) // hparams.tacotron_batch_size
		if hparams.tacotron_test_size is None:
			assert hparams.tacotron_test_batches == self.test_steps

		with tf.device('/cpu:0'):
			# Create placeholders for inputs and targets. Don't specify batch size because we want
			# to be able to feed different batch sizes at eval time.
			self._placeholders = [
			tf.placeholder(tf.int32, shape=(None, None), name='inputs'),
			tf.placeholder(tf.int32, shape=(None, ), name='input_lengths'),
			tf.placeholder(tf.float32, shape=(None, None, hparams.num_mels), name='mel_targets'),
			tf.placeholder(tf.float32, shape=(None, None), name='token_targets'),
			tf.placeholder(tf.int32, shape=(None, ), name='targets_lengths'),
			]
			queue = tf.FIFOQueue(8, [tf.int32, tf.int32, tf.float32, tf.float32, tf.int32], name='input_queue')
			self._enqueue_op = queue.enqueue(self._placeholders)
			self.inputs, self.input_lengths, self.mel_targets, self.token_targets, self.targets_lengths = queue.dequeue()

			self.inputs.set_shape(self._placeholders[0].shape)
			self.input_lengths.set_shape(self._placeholders[1].shape)
			self.mel_targets.set_shape(self._placeholders[2].shape)
			self.token_targets.set_shape(self._placeholders[3].shape)
			self.targets_lengths.set_shape(self._placeholders[4].shape)

			# Create eval queue for buffering eval data
			eval_queue = tf.FIFOQueue(1, [tf.int32, tf.int32, tf.float32, tf.float32, tf.int32], name='eval_queue')
			self._eval_enqueue_op = eval_queue.enqueue(self._placeholders)
			self.eval_inputs, self.eval_input_lengths, self.eval_mel_targets, self.eval_token_targets, self.eval_targets_lengths = eval_queue.dequeue()

			self.eval_inputs.set_shape(self._placeholders[0].shape)
			self.eval_input_lengths.set_shape(self._placeholders[1].shape)
			self.eval_mel_targets.set_shape(self._placeholders[2].shape)
			self.eval_token_targets.set_shape(self._placeholders[3].shape)
			self.eval_targets_lengths.set_shape(self._placeholders[4].shape)

	def start_threads(self, session):
		self._session = session
		thread = threading.Thread(name='background', target=self._enqueue_next_train_group)
		thread.daemon = True #Thread will close when parent quits
		thread.start()

		thread = threading.Thread(name='background', target=self._enqueue_next_test_group)
		thread.daemon = True #Thread will close when parent quits
		thread.start()

	def _get_test_groups(self, speaker_id):
		meta = self._test_meta[speaker_id][self._test_offset[speaker_id]]
		mel_target = np.load(os.path.join(self._mel_dirs[speaker_id], meta[0])).T
		self._test_offset[speaker_id] += 1
		text = meta[3]
		input_data = np.asarray(text_to_sequence(text, speaker_id, self._cleaner_names), dtype=np.int32)
		#Create parallel sequences containing zeros to represent a non finished sequence
		token_target = np.asarray([0.] * (len(mel_target) - 1))
		return (input_data, mel_target, token_target, len(mel_target))

	def make_test_batches(self):
		# Test on entire test set
		# number of speakers must be integer multiple of batch size
		examples = []
		start = time.time()
		n = self._hparams.tacotron_batch_size
		for i in range(self.speaker_num):
			examples.extend([self._get_test_groups(i) for j in range(len(self._test_meta[i]))])
		examples.sort(key=lambda x: x[-1])
		# Bucket examples based on similar output sequence length for efficiency
		batches = [examples[i: i+n] for j in range(0, len(examples), n)]
		np.random.shuffle(batches)
		end = time.time() - start
		log(f'Generated {len(batches)} test batches of size {n} in {end:.3f} sec')
		return batches

	def _enqueue_next_train_group(self):
		while not self._coord.should_stop():
			start = time.time()

			# Read a group of examples
			examples = []
			n = self._hparams.tacotron_batch_size
			for i in range(self.speaker_num):
				for j in range(n // self.speaker_num * _batches_per_group):
					examples.append(self._get_next_example(i))
			examples.sort(key=lambda x: x[-1])
			# Bucket examples based on similar output sequence length for efficiency
			batches = [examples[i: i+n] for i in range(0, len(examples), n)]
			np.random.shuffle(batches)
			end = time.time() - start
			log(f'Generated {len(batches)} train batches of size {n} in {end:.3f} sec')
			for batch in batches:
				feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, self._hparams.outputs_per_step)))
				self._session.run(self._enqueue_op, feed_dict=feed_dict)

	def _enqueue_next_test_group(self):
		#Create test batches once and evaluate on them for all test steps
		test_batches = self.make_test_batches()
		while not self._coord.should_stop():
			for batch in test_batches:
				feed_dict = dict(zip(self._placeholders, self._prepare_batch(batch, self._hparams.outputs_per_step)))
				self._session.run(self._eval_enqueue_op, feed_dict=feed_dict)

	def _get_next_example(self, speaker_id):
		"""
		Gets a single example (input, mel_target, token_target, mel_length) from_ disk
		"""
		if self._train_offset[speaker_id] >= len(self._train_meta[speaker_id]):
			self._train_offset[speaker_id] = 0
			np.random.shuffle(self._train_meta[speaker_id])
		meta = self._train_meta[speaker_id][self._train_offset[speaker_id]]
		self._train_offset[speaker_id] += 1
		mel_target = np.load(os.path.join(self._mel_dirs[speaker_id], meta[0])).T

		text = meta[3]
		input_data = np.asarray(text_to_sequence(text, speaker_id, self._cleaner_names), dtype=np.int32)
		#Create parallel sequences containing zeros to represent a non finished sequence
		token_target = np.asarray([0.] * (len(mel_target) - 1))

		return (input_data, mel_target, token_target, len(mel_target))

	def _prepare_batch(self, batch, outputs_per_step):
		inputs = self._prepare_inputs([x[0] for x in batch])
		input_lengths = np.asarray([len(x[0]) for x in batch], dtype=np.int32)
		mel_targets = self._prepare_targets([x[1] for x in batch], outputs_per_step)
		#Pad sequences with 1 to infer that the sequence is done
		token_targets = self._prepare_token_targets([x[2] for x in batch], outputs_per_step)
		targets_lengths = np.asarray([x[3] for x in batch], dtype=np.int32) #Used to mask loss
		return (inputs, input_lengths, mel_targets, token_targets, targets_lengths)

	def _prepare_inputs(self, inputs):
		max_len = max([len(x) for x in inputs])
		return np.stack([self._pad_input(x, max_len) for x in inputs])

	def _prepare_targets(self, targets, alignment):
		max_len = max([len(t) for t in targets])
		return np.stack([self._pad_target(t, self._round_up(max_len, alignment)) for t in targets])

	def _prepare_token_targets(self, targets, alignment):
		max_len = max([len(t) for t in targets]) + 1
		return np.stack([self._pad_token_target(t, self._round_up(max_len, alignment)) for t in targets])

	def _pad_input(self, x, length):
		return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

	def _pad_target(self, t, length):
		return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=self._target_pad)

	def _pad_token_target(self, t, length):
		return np.pad(t, (0, length - t.shape[0]), mode='constant', constant_values=self._token_pad)

	def _round_up(self, x, multiple):
		remainder = x % multiple
		return x if remainder == 0 else x + multiple - remainder

	def _round_down(self, x, multiple):
		remainder = x % multiple
		return x if remainder == 0 else x - remainder

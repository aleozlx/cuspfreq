import numpy as np
import tensorflow as tf

spfreqOp = tf.load_op_library("./spfreq.so").superpixel_freq

def standard(segments, conv_map_shape, N_segments):
	subsample = tuple(np.array(segments.shape) // np.array(conv_map_shape))
	from skimage.util import view_as_blocks
	W = segments[np.newaxis,...] == np.arange(N_segments)[...,np.newaxis,np.newaxis]
	W = np.sum(view_as_blocks(W, (1,)+subsample), axis=(3,4,5))
	return W.astype(float)

class SpfreqKernelTest(tf.test.TestCase):
	def zero_chunked(self, batch, nsp, cm, pool = 16):
		sample_input = np.random.randint(nsp, size = (batch, cm*pool, cm*pool)).astype('i4')
		sample_output = np.zeros((batch, nsp, cm, cm))
		with self.test_session():
			result = spfreqOp(sample_input, output_shape = (nsp, cm, cm), test_kernel=12).eval()
			self.assertAllEqual(result, sample_output)

	def testZero_chunked_00(self):
		self.zero_chunked(1, 3, 2, 2)

	def testZero_chunked_01_single_block(self):
		self.zero_chunked(1, 4, 16)

	def testZero_chunked_02_cm_blocks(self):
		self.zero_chunked(1, 4, 20)

	def testZero_chunked_03_sp_blocks(self):
		self.zero_chunked(1, 8, 16)

	def testZero_chunked_04_spcm_blocks(self):
		self.zero_chunked(1, 8, 20)

	def testZero_chunked_05_batches(self):
		self.zero_chunked(3, 4, 16)

	def testZero_chunked_06_more_batches(self):
		self.zero_chunked(20, 4, 16)

	def testZero_chunked_07_batches_sp(self):
		self.zero_chunked(20, 8, 16)

	def testZero_chunked_08_full(self):
		self.zero_chunked(32, 600, 64)

	def incr_chunked(self, batch, nsp, cm, pool = 16):
		sample_input = np.random.randint(nsp, size = (batch, cm*pool, cm*pool)).astype('i4')
		sample_output = np.ones((batch, nsp, cm, cm))
		with self.test_session():
			result = spfreqOp(sample_input, output_shape = (nsp, cm, cm), test_kernel=13).eval()
			self.assertAllEqual(result, sample_output)

	def testIncr_chunked_00(self):
		self.incr_chunked(1, 3, 2, 2)

	def testIncr_chunked_01_single_block(self):
		self.incr_chunked(1, 4, 16)

	def testIncr_chunked_02_cm_blocks(self):
		self.incr_chunked(1, 4, 20)

	def testIncr_chunked_03_sp_blocks(self):
		self.incr_chunked(1, 8, 16)

	def testIncr_chunked_04_spcm_blocks(self):
		self.incr_chunked(1, 8, 20)

	def testIncr_chunked_05_batches(self):
		self.incr_chunked(3, 4, 16)

	def testIncr_chunked_06_more_batches(self):
		self.incr_chunked(20, 4, 16)

	def testIncr_chunked_07_batches_sp(self):
		self.incr_chunked(20, 8, 16)

	def testIncr_chunked_08_full(self):
		self.incr_chunked(32, 600, 64)

	# def testAreaChunked(self):
	# 	with self.test_session():
	# 		result = spfreqOp(sample_input, output_shape = (4, 2, 2), test_kernel=11)
	# 		self.assertAllEqual(result.eval(), sample_area)
	
	def testAreaChunked_adim(self):
		batch = 1; nsp = 3; cm = 4
		sample_input = np.random.randint(nsp, size = (batch, cm*2, cm*2)).astype('i4')
		sample_area = np.array([standard(segments, (cm, cm), nsp) for segments in sample_input])
		with self.test_session():
			result = spfreqOp(sample_input, output_shape = (nsp, cm, cm), test_kernel=11)
			result = result.eval()
			# print('result', result)
			# print('answer', sample_area)
			print('diff', sample_area-result)
			self.assertAllEqual(result, sample_area)

if __name__ == "__main__":
	tf.test.main()
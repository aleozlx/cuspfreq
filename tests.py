import numpy as np
import tensorflow as tf

spfreqOp = tf.load_op_library("./spfreq.so").superpixel_freq

def _segments2area(segments, conv_map_shape, nsp):
	N_segments = nsp
	subsample = tuple(np.array(segments.shape) // np.array(conv_map_shape))
	from skimage.util import view_as_blocks
	W = segments[np.newaxis,...] == np.arange(N_segments)[...,np.newaxis,np.newaxis]
	W = np.sum(view_as_blocks(W, (1,)+subsample), axis=(3,4,5))
	return W.astype(float)

class SpfreqKernelTest(tf.test.TestCase):
	# def testAreaChunked(self):
	# 	with self.test_session():
	# 		result = spfreqOp(sample_input, output_shape = (4, 2, 2), test_kernel=11)
	# 		self.assertAllEqual(result.eval(), sample_area)

	def testAreaChunked_adim(self):
		pass
		# nsp = 3; cm = 4
		# sample_input = np.random.randint(nsp, size = (2, cm*2, cm*2)).astype('i4')
		# sample_area = np.array([_segments2area(segments, (cm, cm), nsp) for segments in sample_input])
		# with self.test_session():
		# 	result = spfreqOp(sample_input, output_shape = (nsp, cm, cm), test_kernel=11)
		# 	result = result.eval()
		# 	# print('result', result)
		# 	# print('answer', sample_area)
		# 	print('diff', sample_area-result)
		# 	self.assertAllEqual(result, sample_area)

	def testIncrChunked_adim(self):
		batch = 32; nsp = 600; cm = 64
		sample_input = np.random.randint(nsp, size = (batch, cm*16, cm*16)).astype('i4')
		sample_area = np.ones((batch, nsp, cm, cm))
		with self.test_session():
			result = spfreqOp(sample_input, output_shape = (nsp, cm, cm), test_kernel=13).eval()
			# print('diff', np.sum((sample_area-result)**2, axis = (1,2,3)))
			self.assertAllEqual(result, sample_area)

if __name__ == "__main__":
	tf.test.main()
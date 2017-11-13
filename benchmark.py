import os, sys
import time, logging
import numpy as np
import tensorflow as tf

logging.basicConfig(stream=sys.stderr)
logging.getLogger("benchmark").setLevel(logging.INFO)

spfreqOp = tf.load_op_library("./spfreq.so").superpixel_freq

def log_timer(d0, d1):
	_round = lambda d: round(d*1000,2)
	logging.getLogger("benchmark").info('CPU {}ms  GPU {}ms'.format(_round(d0), _round(d1)))

def standard(segments, conv_map_shape, N_segments):
	subsample = tuple(np.array(segments.shape) // np.array(conv_map_shape))
	from skimage.util import view_as_blocks
	W = segments[np.newaxis,...] == np.arange(N_segments)[...,np.newaxis,np.newaxis]
	W = np.sum(view_as_blocks(W, (1,)+subsample), axis=(3,4,5))
	return W.astype(float)

class SpfreqKernelTest(tf.test.TestCase):
	def area_chunked(self, batch, nsp, cm, pool = 16):
			sample_input = np.random.randint(nsp, size = (batch, cm*2, cm*2)).astype('i4')
			t0 = time.time()
			sample_area = np.array([standard(segments, (cm, cm), nsp) for segments in sample_input])
			t1 = time.time()
			with self.test_session():
				result = spfreqOp(sample_input, output_shape = (nsp, cm, cm), test_kernel=11).eval()
				t2 = time.time()
				self.assertAllEqual(result, sample_area)
			log_timer(t1-t0, t2-t1)

	def testArea_chunked_02_cm_blocks(self):
		self.area_chunked(16, 600, 20)

	def testArea_chunked_02_cm_blocks1(self):
		self.area_chunked(16, 600, 40)

	def testArea_chunked_02_cm_blocks2(self):
		self.area_chunked(16, 600, 80)

	def testArea_chunked_03_sp_blocks(self):
		self.area_chunked(16, 200, 64)

	def testArea_chunked_03_sp_blocks1(self):
		self.area_chunked(16, 400, 64)

	def testArea_chunked_03_sp_blocks2(self):
		self.area_chunked(16, 600, 64)

	def testArea_chunked_03_sp_blocks3(self):
		self.area_chunked(16, 800, 64)

	def testArea_chunked_05_batches(self):
		self.area_chunked(16, 600, 64)

	def testArea_chunked_06_more_batches(self):
		self.area_chunked(20, 600, 64)

	def testArea_chunked_07_batches_sp(self):
		self.area_chunked(32, 600, 64)

if __name__ == "__main__":
	tf.test.main()

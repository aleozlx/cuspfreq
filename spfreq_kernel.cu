/*
   Copyright 2017 Alex Yang

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "spfreq.h"
// #include <iostream>
// #include <cstdio>
#ifdef GOOGLE_CUDA
#include <algorithm>
#include <cublas_v2.h>

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS
#define GDIV(a,b) (((a)+(b)-1)/(b))

#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

namespace spfreq {

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
	if (code != cudaSuccess)
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
}

__device__ __constant__ SuperpixelFreqShape _spfreq_shape;

// #define INTERIOR3D (idx_batch < spfreq_shape.batch_size && idx_superpixel < spfreq_shape.num_superpixel && idx_spatial < spfreq_shape.spatial_size)

// Base data sturcture for input/output tensors' metadata
struct SuperpixelFreqMetaBase {
	const SuperpixelFreqShape *shape;
	const int idx_spatial;
	__device__ SuperpixelFreqMetaBase(const SuperpixelFreqShape *spfreq_shape)
		: shape(spfreq_shape),
		  idx_spatial(blockDim.y * blockIdx.y + threadIdx.y) {}
};

// Address resolving for input tensor representing superpixel segments, shaped (size_batch, size_rows, size_columns)
struct SuperpixelSegmentsMeta : SuperpixelFreqMetaBase {
	const int idx_tile_rows, idx_tile_cols;
	__device__ SuperpixelSegmentsMeta(const SuperpixelFreqShape *spfreq_shape)
		: SuperpixelFreqMetaBase(spfreq_shape),
		  idx_tile_rows(idx_spatial / shape->spatial.cols),
		  idx_tile_cols(idx_spatial % shape->spatial.cols) {}

	// Get relative input address assigned to the thread
	__device__ int addr(int idx_batch, int thread_i, int thread_j) const {
		int i = idx_tile_rows * shape->per_thread.rows + thread_i,
			j = idx_tile_cols * shape->per_thread.cols + thread_j;
		return idx_batch * shape->stride.in[0] + i * shape->stride.in[1] + j;
	}
};

// Address resolving for output tensor representing superpixel frequencies/area, shaped (size_batch, size_superpixels, size_spatial)
struct SuperpixelFreqMeta : SuperpixelFreqMetaBase {
	const int idx_superpixel;
	__device__ SuperpixelFreqMeta(const SuperpixelFreqShape *spfreq_shape, int offset_superpixel)
		: SuperpixelFreqMetaBase(spfreq_shape),
		  idx_superpixel(blockDim.x * blockIdx.x + threadIdx.x + offset_superpixel) {}
	
	// Returns whether the CUDA thread got assigned to a point within the boundary of output space.
	__device__ bool interior() const {
		return idx_superpixel < shape->num_superpixel && idx_spatial < shape->spatial_size;
	}

	// Get relative output address assigned to the thread
	__device__ int addr(int idx_batch) const {
		return idx_batch * shape->stride.out[0] + idx_superpixel * shape->stride.out[1] + idx_spatial;
	}
};

template <typename T_in, typename T_out>
__global__ void SuperpixelFreqKernel_11Area_chunked(const T_in* data_in, T_out* data_out, int idx_batch, int offset_superpixel) {
	const SuperpixelFreqShape spfreq_shape = _spfreq_shape;
	const SuperpixelSegmentsMeta _in(&spfreq_shape);
	const SuperpixelFreqMeta _out(&spfreq_shape, offset_superpixel);
	if(_out.interior()){
		T_in sum = 0;
		for(int i = 0; i<spfreq_shape.per_thread.rows; ++i) for(int j = 0; j<spfreq_shape.per_thread.cols; ++j)
			if(data_in[_in.addr(idx_batch, i, j)] == _out.idx_superpixel) sum += 1;
		data_out[_out.addr(idx_batch)] = static_cast<T_out>(sum);
	}
}

// CUDA kernels for debugging purposes
template <typename T_in, typename T_out>
__global__ void SuperpixelFreqKernel_12Zero_chunked(const T_in* data_in, T_out* data_out, int idx_batch, int offset_superpixel) {
	const SuperpixelFreqShape spfreq_shape = _spfreq_shape;
	const SuperpixelFreqMeta _out(&spfreq_shape, offset_superpixel);
	if(_out.interior()) data_out[_out.addr(idx_batch)] = 0;
}

template <typename T_in, typename T_out>
__global__ void SuperpixelFreqKernel_13Incr_chunked(const T_in* data_in, T_out* data_out, int idx_batch, int offset_superpixel) {
	const SuperpixelFreqShape spfreq_shape = _spfreq_shape;
	const SuperpixelFreqMeta _out(&spfreq_shape, offset_superpixel);
	if(_out.interior()) data_out[_out.addr(idx_batch)] += 1;
}

template <typename T_in, typename T_out>
__global__ void SuperpixelFreqKernel_14SpIdx_chunked(const T_in* data_in, T_out* data_out, int idx_batch, int offset_superpixel) {
	const SuperpixelFreqShape spfreq_shape = _spfreq_shape;
	SuperpixelFreqMeta _out(&spfreq_shape, offset_superpixel);
	if(_out.interior()) data_out[_out.addr(idx_batch)] = static_cast<T_out>(_out.idx_superpixel);
}

template <typename T>
T* gpu_vec_ones(int n, cudaStream_t stream){
	T ones[n], *d_ones;
	std::fill_n(ones, n, static_cast<T>(1));
	cudaCheck(cudaMalloc(&d_ones, n*sizeof(T)));
	cudaMemcpyAsync(d_ones, ones, n*sizeof(T), cudaMemcpyHostToDevice, stream);
	return d_ones;
}

// template <typename T_out>
// __global__ void SuperpixelFreqKernel_21Norm(T_out* data_out, T_out* rsum) {
// 	const SuperpixelFreqShape spfreq_shape = _spfreq_shape;
// 	const int idx_batch = blockIdx.z,
// 			  idx_superpixel = blockDim.x * blockIdx.x + threadIdx.x,
// 			  idx_spatial = blockDim.y * blockIdx.y + threadIdx.y;
// 	if(INTERIOR3D){
// 		T_out normalization = rsum[idx_batch * spfreq_shape.num_superpixel + idx_superpixel];
// 		if(normalization!=0.0)
// 			data_out[idx_batch * spfreq_shape.stride.out[0] + 
// 				idx_superpixel * spfreq_shape.stride.out[1] + idx_spatial] /= normalization;
// 	}
// }

template <typename T_in, typename T_out>
void unit_test(int test_case, const GPUDevice& device, const SuperpixelFreqShape &shape, const T_in* data_in, T_out* data_out){
	dim3 blk_pool(4, 256, 1),
		grid_pool(GDIV(shape.num_superpixel, blk_pool.x),
			GDIV(shape.per_thread.rows * shape.per_thread.cols, blk_pool.y),
			GDIV(shape.batch_size, blk_pool.z));

#define TEST_CASE_POOL(Kernel) Kernel <T_in, T_out> <<<grid_pool, blk_pool, 0, device.stream()>>> (data_in, data_out)
#define SPFREQ_KERNEL(K) K <T_in, T_out> <<<grid_chunk, blk_pool, 0, device.stream()>>> (data_in, data_out, p_batch, p_sp)

	switch(test_case){
		case 11:
			{
				dim3 grid_chunk(1, GDIV(shape.spatial_size, blk_pool.y), 1);
				for(int p_batch = 0; p_batch < shape.batch_size; ++p_batch){
					for(int p_sp=0; p_sp<shape.num_superpixel; p_sp += blk_pool.x)
						SPFREQ_KERNEL(SuperpixelFreqKernel_11Area_chunked);
				}
			}
			break;
		case 12:
			{
				dim3 grid_chunk(1, GDIV(shape.spatial_size, blk_pool.y), 1);
				for(int p_batch = 0; p_batch < shape.batch_size; ++p_batch){
					for(int p_sp=0; p_sp<shape.num_superpixel; p_sp += blk_pool.x)
						SPFREQ_KERNEL(SuperpixelFreqKernel_12Zero_chunked);
				}
			}
			break;
		case 13:
			{
				dim3 grid_chunk(1, GDIV(shape.spatial_size, blk_pool.y), 1);
				for(int p_batch = 0; p_batch < shape.batch_size; ++p_batch){
					for(int p_sp=0; p_sp<shape.num_superpixel; p_sp += blk_pool.x)
						SPFREQ_KERNEL(SuperpixelFreqKernel_12Zero_chunked);
					for(int p_sp=0; p_sp<shape.num_superpixel; p_sp += blk_pool.x)
						SPFREQ_KERNEL(SuperpixelFreqKernel_13Incr_chunked);
				}
			}
			break;
		case 14:
			{
				dim3 grid_chunk(1, GDIV(shape.spatial_size, blk_pool.y), 1);
				for(int p_batch = 0; p_batch < shape.batch_size; ++p_batch){
					for(int p_sp=0; p_sp<shape.num_superpixel; p_sp += blk_pool.x)
						SPFREQ_KERNEL(SuperpixelFreqKernel_14SpIdx_chunked);
				}
			}
			break;
		// case 20:
		// 	{
		// 		TEST_CASE_POOL(SuperpixelFreqKernel_10Area);
		// 		T_out* rsum = SuperpixelFreqKernel_20RSum(device, shape, data_out);
		// 		cudaCheck(cudaMemcpyAsync(data_out, rsum, shape.batch_size*shape.num_superpixel*sizeof(T_out), cudaMemcpyDeviceToDevice, device.stream()));
		// 		cudaFree(rsum);
		// 	}
		// 	break;
		// case 21:
		// 	{
		// 		TEST_CASE_POOL(SuperpixelFreqKernel_10Area);
		// 		T_out* rsum = SuperpixelFreqKernel_20RSum(device, shape, data_out);
		// 		SuperpixelFreqKernel_21Norm <T_out> <<<grid_pool, blk_pool, 0, device.stream()>>> (data_out, rsum);
		// 		cudaFree(rsum);
		// 	}
		// 	break;
	}
	cudaStreamSynchronize(device.stream());
}


template <typename T_in, typename T_out>
struct SuperpixelFreqFunctor<GPUDevice, T_in, T_out> {
	void operator()(const GPUDevice& device, SuperpixelFreqShape shape, const T_in* data_in, T_out* data_out, const int test_kernel = -1) {
		dim3 /* input shape: (batch_size, num_superpixel, {in_rows, in_cols})
			    output shape: (batch_size, num_superpixel, spatial = {out_rows, out_cols})
			    thread assignments: (x, y, z) = 
			    	(num_superpixel, per_thread_tile_size = {in_rows/out_rows, in_cols/out_cols}, batch_size) -> (num_superpixel, spatial, batch_size)
			 */
			blk_pool(4, 256, 1), 
			grid_pool(GDIV(shape.num_superpixel, blk_pool.x),
				GDIV(shape.per_thread.rows * shape.per_thread.cols, blk_pool.y),
				GDIV(shape.batch_size, blk_pool.z)),

			/* 	input shape: (batch_size, num_superpixel, spatial), (batch_size, num_superpixel)
				output shape: (batch_size, num_superpixel, spatial)
				thread assignments: (x, y, z) = (num_superpixel, spatial, batch_size) -> (num_superpixel, spatial, batch_size)
			*/
			blk_norm(512, 1),
			grid_norm(GDIV(shape.num_superpixel, blk_norm.x), GDIV(shape.batch_size, blk_norm.y));

		dim3 grid_chunk(1, GDIV(shape.spatial_size, blk_pool.y), 1);

		// Transfer constants
		cudaCheck(cudaMemcpyToSymbolAsync(_spfreq_shape, &shape, sizeof(shape), 0,
			cudaMemcpyHostToDevice, device.stream()));

		// Unit test trap
		if(test_kernel >= 0) { unit_test(test_kernel, device, shape, data_in, data_out); return; }

		// Op begins
		for(int p_batch = 0; p_batch < shape.batch_size; ++p_batch){
			for(int p_sp=0; p_sp<shape.num_superpixel; p_sp += blk_pool.x)
				SPFREQ_KERNEL(SuperpixelFreqKernel_11Area_chunked);
		}
		T_out* rsum = GetL1Norm(device, shape, data_out);
		// SuperpixelFreqKernel_21Norm <T_out> <<<grid_pool, blk_pool, 0, device.stream()>>> (data_out, rsum);
		cudaFree(rsum);
	}

	T_out* GetL1Norm(const GPUDevice& device, const SuperpixelFreqShape &shape, const T_out* data_out){
		T_out *rsum; cudaCheck(cudaMalloc(&rsum, shape.batch_size*shape.num_superpixel*sizeof(T_out)));
		cublasHandle_t handle; cublasCreate(&handle); cublasSetStream(handle, device.stream());
		const float coef[] = {1.0f, 0.0f};
		float *ones = gpu_vec_ones<float>(shape.num_superpixel, device.stream());
		for(int idx_batch=0; idx_batch<shape.batch_size; ++idx_batch){
			T_out* d_idata = const_cast<T_out*>(&data_out[idx_batch * shape.stride.out[0]]);
			T_out* d_odata = &rsum[idx_batch * shape.num_superpixel];
			cublasSgemv(handle,
				CUBLAS_OP_T, shape.spatial_size, shape.num_superpixel, coef+0, d_idata, shape.spatial_size,
				ones, 1, coef+1,
				d_odata, 1);
		}
		cublasDestroy(handle);
		cudaFree(ones);
		return rsum;
	}
};

typedef Eigen::GpuDevice GPUDevice;
template struct SuperpixelFreqFunctor<GPUDevice, int32, float>;

}

#endif  // GOOGLE_CUDA

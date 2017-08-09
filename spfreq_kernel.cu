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

#include <iostream>
#include <cstdio>
#ifdef GOOGLE_CUDA
#include "spfreq.h"
#include <cublas_v2.h>
#include <algorithm>

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

#define INTERIOR3D (IDX_BATCH < spfreq_shape.batch_sz && IDX_SUPERPIXEL < spfreq_shape.nsp && IDX_SPATIAL < spfreq_shape.spatial_sz)

template <typename T_in, typename T_out>
__global__ void SuperpixelFreqKernel_01SuperpixelIndex(const T_in* data_in, T_out* data_out) {
	const SuperpixelFreqShape spfreq_shape = _spfreq_shape;
	const int IDX_BATCH = blockIdx.z,
			  IDX_SUPERPIXEL = blockDim.x * blockIdx.x + threadIdx.x,
			  IDX_SPATIAL = blockDim.y * blockIdx.y + threadIdx.y;
	T_out &output = data_out[IDX_BATCH * spfreq_shape.stride.out[0] + 
						IDX_SUPERPIXEL * spfreq_shape.stride.out[1] + IDX_SPATIAL];
	if(INTERIOR3D) output = static_cast<T_out>(IDX_SUPERPIXEL);
}

template <typename T_in, typename T_out>
__global__ void SuperpixelFreqKernel_10Area(const T_in* data_in, T_out* data_out) {
	const SuperpixelFreqShape spfreq_shape = _spfreq_shape;
	const int IDX_BATCH = blockIdx.z,
			  IDX_SUPERPIXEL = blockDim.x * blockIdx.x + threadIdx.x,
			  IDX_SPATIAL = blockDim.y * blockIdx.y + threadIdx.y;
	if(INTERIOR3D){
		const int PER_THREAD_ROWS = spfreq_shape.per_thread.rows,
			  PER_THREAD_COLS = spfreq_shape.per_thread.cols,
			  IDX_TILE_ROWS = threadIdx.y / PER_THREAD_COLS * spfreq_shape.spatial.rows,
			  IDX_TILE_COLS = threadIdx.y % PER_THREAD_COLS * spfreq_shape.spatial.cols,
			  THREAD_INPUT_OFFSET = IDX_BATCH * spfreq_shape.stride.in[0],
			  INPUT_SPATIAL_STRIDE = spfreq_shape.stride.in[1];
		T_in sum = 0;
		for(int i = 0; i<PER_THREAD_ROWS; ++i) for(int j = 0; j<PER_THREAD_COLS; ++j) {
			T_in input_val = data_in[THREAD_INPUT_OFFSET + (IDX_TILE_ROWS+i) * INPUT_SPATIAL_STRIDE + (IDX_TILE_COLS+j)];
			if(input_val == IDX_SUPERPIXEL) sum += 1;
		}
		data_out[IDX_BATCH * spfreq_shape.stride.out[0] + 
			IDX_SUPERPIXEL * spfreq_shape.stride.out[1] + IDX_SPATIAL] = static_cast<T_out>(sum);
	}
}

#define ARRAY_OUT(N,S,P) data_out[(N)*spfreq_shape.stride.out[0]+(S)*spfreq_shape.stride.out[1]+(P)]

struct meta_t {
	SuperpixelFreqShape const *shape; 
	int IDX_SPATIAL;
	__device__ meta_t(const SuperpixelFreqShape *spfreq_shape) {
		this->shape = spfreq_shape;
		this->IDX_SPATIAL = blockDim.y * blockIdx.y + threadIdx.y;
	}
};

struct meta_in_t : meta_t {
	int IDX_TILE_ROWS, IDX_TILE_COLS;
	__device__ meta_in_t(const SuperpixelFreqShape *spfreq_shape) : meta_t(spfreq_shape) {
		this->IDX_TILE_ROWS = this->IDX_SPATIAL / shape->spatial.cols,
		this->IDX_TILE_COLS = this->IDX_SPATIAL % shape->spatial.cols;
	}
	__device__ int addr(int n, int thread_i, int thread_j){
		int i = this->IDX_TILE_ROWS * shape->per_thread.rows + thread_i,
			j = this->IDX_TILE_COLS * shape->per_thread.cols + thread_j;
		return n * shape->stride.in[0] + i * shape->stride.in[1] + j;
	}
};

struct meta_out_t : meta_t {
	int IDX_SUPERPIXEL;
	__device__ meta_out_t(const SuperpixelFreqShape *spfreq_shape, int OFFSET_SUPERPIXEL) : meta_t(spfreq_shape) {
		this->IDX_SUPERPIXEL = blockDim.x * blockIdx.x + threadIdx.x + OFFSET_SUPERPIXEL;
	}
	 __device__ bool interior() {
		return this->IDX_SUPERPIXEL < shape->nsp && this->IDX_SPATIAL < shape->spatial_sz;
	}
	__device__ int addr(int n){
		return n * shape->stride.out[0] + this->IDX_SUPERPIXEL * shape->stride.out[1] + this->IDX_SPATIAL;
	}
};

template <typename T_in, typename T_out>
__global__ void SuperpixelFreqKernel_11Area_chunked(const T_in* data_in, T_out* data_out, int IDX_BATCH, int OFFSET_SUPERPIXEL) {
	const SuperpixelFreqShape spfreq_shape = _spfreq_shape;
	meta_in_t _in(&spfreq_shape); meta_out_t _out(&spfreq_shape, OFFSET_SUPERPIXEL);
	if(_out.interior()){
		T_in sum = 0;
		for(int i = 0; i<spfreq_shape.per_thread.rows; ++i) for(int j = 0; j<spfreq_shape.per_thread.cols; ++j)
			if(data_in[_in.addr(IDX_BATCH, i, j)] == _out.IDX_SUPERPIXEL) sum += 1;
		data_out[_out.addr(IDX_BATCH)] = static_cast<T_out>(sum);
	}
}

template <typename T_in, typename T_out>
__global__ void SuperpixelFreqKernel_12Zero_chunked(const T_in* data_in, T_out* data_out, int IDX_BATCH, int OFFSET_SUPERPIXEL) {
	const SuperpixelFreqShape spfreq_shape = _spfreq_shape;
	meta_out_t _out(&spfreq_shape, OFFSET_SUPERPIXEL);
	if(_out.interior()) data_out[_out.addr(IDX_BATCH)] = 0;
}

template <typename T_in, typename T_out>
__global__ void SuperpixelFreqKernel_13Incr_chunked(const T_in* data_in, T_out* data_out, int IDX_BATCH, int OFFSET_SUPERPIXEL) {
	const SuperpixelFreqShape spfreq_shape = _spfreq_shape;
	meta_out_t _out(&spfreq_shape, OFFSET_SUPERPIXEL);
	if(_out.interior()) data_out[_out.addr(IDX_BATCH)] += 1;
}

template <typename T_in, typename T_out>
__global__ void SuperpixelFreqKernel_14SpIdx_chunked(const T_in* data_in, T_out* data_out, int IDX_BATCH, int OFFSET_SUPERPIXEL) {
	const SuperpixelFreqShape spfreq_shape = _spfreq_shape;
	meta_out_t _out(&spfreq_shape, OFFSET_SUPERPIXEL);
	if(_out.interior()) data_out[_out.addr(IDX_BATCH)] = static_cast<T_out>(_out.IDX_SUPERPIXEL);
}

template <typename T_out>
__global__ void SuperpixelFreqKernel_21Norm(T_out* data_out, T_out* rsum) {
	const SuperpixelFreqShape spfreq_shape = _spfreq_shape;
	const int IDX_BATCH = blockIdx.z,
			  IDX_SUPERPIXEL = blockDim.x * blockIdx.x + threadIdx.x,
			  IDX_SPATIAL = blockDim.y * blockIdx.y + threadIdx.y;
	if(INTERIOR3D){
		T_out normalization = rsum[IDX_BATCH * spfreq_shape.nsp + IDX_SUPERPIXEL];
		if(normalization!=0.0)
			data_out[IDX_BATCH * spfreq_shape.stride.out[0] + 
				IDX_SUPERPIXEL * spfreq_shape.stride.out[1] + IDX_SPATIAL] /= normalization;
	}
}

template <typename T>
T* gpu_vec_ones(int n, cudaStream_t stream){
	T ones[n], *d_ones;
	std::fill_n(ones, n, static_cast<T>(1));
	cudaCheck(cudaMalloc(&d_ones, n*sizeof(T)));
	cudaMemcpyAsync(d_ones, ones, n*sizeof(T), cudaMemcpyHostToDevice, stream);
	return d_ones;
}

template<typename T_out>
T_out* SuperpixelFreqKernel_20RSum(const GPUDevice& device, const SuperpixelFreqShape &shape, const T_out* data_out){
	T_out *rsum; cudaCheck(cudaMalloc(&rsum, shape.batch_sz*shape.nsp*sizeof(T_out)));
	cublasHandle_t handle; cublasCreate(&handle); cublasSetStream(handle, device.stream());
	const float coef[] = {1.0f, 0.0f};
	float *ones = gpu_vec_ones<float>(shape.nsp, device.stream());
	for(int IDX_BATCH=0; IDX_BATCH<shape.batch_sz; ++IDX_BATCH){
		T_out* d_idata = const_cast<T_out*>(&data_out[IDX_BATCH * shape.stride.out[0]]);
		T_out* d_odata = &rsum[IDX_BATCH * shape.nsp];
		cublasSgemv(handle,
			CUBLAS_OP_T, shape.spatial_sz, shape.nsp, coef+0, d_idata, shape.spatial_sz,
			ones, 1, coef+1,
			d_odata, 1);
	}
	cublasDestroy(handle);
	cudaFree(ones);
	return rsum;
}

template <typename T_in, typename T_out>
void unit_test(int test_case, const GPUDevice& device, const SuperpixelFreqShape &shape, const T_in* data_in, T_out* data_out){
	dim3 blk_pool(4, 256, 1),
		grid_pool(GDIV(shape.nsp, blk_pool.x),
			GDIV(shape.per_thread.rows * shape.per_thread.cols, blk_pool.y),
			GDIV(shape.batch_sz, blk_pool.z));

#define TEST_CASE_POOL(Kernel) Kernel <T_in, T_out> <<<grid_pool, blk_pool, 0, device.stream()>>> (data_in, data_out)
#define SPFREQ_KERNEL(K) K <T_in, T_out> <<<grid_chunk, blk_pool, 0, device.stream()>>> (data_in, data_out, p_batch, p_sp)

	switch(test_case){
		case 1: TEST_CASE_POOL(SuperpixelFreqKernel_01SuperpixelIndex); break;
		case 10: TEST_CASE_POOL(SuperpixelFreqKernel_10Area); break;
		case 11:
			{
				dim3 grid_chunk(1, GDIV(shape.spatial_sz, blk_pool.y), 1);
				for(int p_batch = 0; p_batch < shape.batch_sz; ++p_batch){
					for(int p_sp=0; p_sp<shape.nsp; p_sp += blk_pool.x)
						SPFREQ_KERNEL(SuperpixelFreqKernel_11Area_chunked);
				}
			}
			break;
		case 12:
			{
				dim3 grid_chunk(1, GDIV(shape.spatial.rows * shape.spatial.cols, blk_pool.y), 1);
				
				for(int p_batch = 0; p_batch < shape.batch_sz; ++p_batch){
					for(int p_sp=0; p_sp<shape.nsp; p_sp += blk_pool.x)
						SPFREQ_KERNEL(SuperpixelFreqKernel_12Zero_chunked);
				}
			}
			break;
		case 13:
			{
				dim3 grid_chunk(1, GDIV(shape.spatial.rows * shape.spatial.cols, blk_pool.y), 1);
				
				for(int p_batch = 0; p_batch < shape.batch_sz; ++p_batch){
					for(int p_sp=0; p_sp<shape.nsp; p_sp += blk_pool.x)
						SPFREQ_KERNEL(SuperpixelFreqKernel_12Zero_chunked);
					for(int p_sp=0; p_sp<shape.nsp; p_sp += blk_pool.x)
						SPFREQ_KERNEL(SuperpixelFreqKernel_13Incr_chunked);
				}
			}
			break;
		case 20:
			{
				TEST_CASE_POOL(SuperpixelFreqKernel_10Area);
				T_out* rsum = SuperpixelFreqKernel_20RSum(device, shape, data_out);
				cudaCheck(cudaMemcpyAsync(data_out, rsum, shape.batch_sz*shape.nsp*sizeof(T_out), cudaMemcpyDeviceToDevice, device.stream()));
				cudaFree(rsum);
			}
			break;
		case 21:
			{
				TEST_CASE_POOL(SuperpixelFreqKernel_10Area);
				T_out* rsum = SuperpixelFreqKernel_20RSum(device, shape, data_out);
				SuperpixelFreqKernel_21Norm <T_out> <<<grid_pool, blk_pool, 0, device.stream()>>> (data_out, rsum);
				cudaFree(rsum);
			}
			break;
	}
	cudaStreamSynchronize(device.stream());
}


template <typename T_in, typename T_out>
struct SuperpixelFreqFunctor<GPUDevice, T_in, T_out> {
	void operator()(const GPUDevice& device, SuperpixelFreqShape shape, const T_in* data_in, T_out* data_out, const int test_kernel = -1) {
		dim3 /* input shape: (batch_sz, nsp, {in_rows, in_cols})
			    output shape: (batch_sz, nsp, spatial = {out_rows, out_cols})
			    thread assignments: (x, y, z) = 
			    	(nsp, per_thread_tile_sz = {in_rows/out_rows, in_cols/out_cols}, batch_sz) -> (nsp, spatial, batch_sz)
			 */
			blk_pool(4, 256, 1), 
			grid_pool(GDIV(shape.nsp, blk_pool.x),
				GDIV(shape.per_thread.rows * shape.per_thread.cols, blk_pool.y),
				GDIV(shape.batch_sz, blk_pool.z)),

			/* 	input shape: (batch_sz, nsp, spatial), (batch_sz, nsp)
				output shape: (batch_sz, nsp, spatial)
				thread assignments: (x, y, z) = (nsp, spatial, batch_sz) -> (nsp, spatial, batch_sz)
			*/
			blk_norm(512, 1),
			grid_norm(GDIV(shape.nsp, blk_norm.x), GDIV(shape.batch_sz, blk_norm.y));

		// Transfer constants
		cudaCheck(cudaMemcpyToSymbolAsync(_spfreq_shape, &shape, sizeof(SuperpixelFreqShape), 0,
			cudaMemcpyHostToDevice, device.stream()));

		// Unit test trap
		if(test_kernel >= 0) { unit_test(test_kernel, device, shape, data_in, data_out); return; }

		// Op begins
		SuperpixelFreqKernel_10Area<T_in, T_out> <<<grid_pool, blk_pool, 0, device.stream()>>> (data_in, data_out);
		T_out* rsum = SuperpixelFreqKernel_20RSum(device, shape, data_out);
		SuperpixelFreqKernel_21Norm <T_out> <<<grid_pool, blk_pool, 0, device.stream()>>> (data_out, rsum);
		cudaFree(rsum);
	}
};

typedef Eigen::GpuDevice GPUDevice;
template struct SuperpixelFreqFunctor<GPUDevice, int32, float>;

}

#endif  // GOOGLE_CUDA

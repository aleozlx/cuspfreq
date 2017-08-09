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

#ifndef _SPFREQ_H_
#define _SPFREQ_H_

#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace spfreq {

template <typename SIZE>
struct SuperpixelFreqShapeBase {
	SIZE nsp, spatial_sz, batch_sz;
	struct { SIZE rows, cols; } _spatial_sz;
	struct { SIZE out[2], in[4]; } stride;
	SIZE temp_sz[4];

	inline SuperpixelFreqShapeBase() { }

	SuperpixelFreqShapeBase(const tensorflow::TensorShape &input_shape, const tensorflow::TensorShape &output_shape) {
		this->batch_sz = input_shape.dim_size(0);
		this->nsp = output_shape.dim_size(0);
		this->_spatial_sz.rows = output_shape.dim_size(1);
		this->_spatial_sz.cols = output_shape.dim_size(2);
		this->spatial_sz = output_shape.dim_size(1) * output_shape.dim_size(2);
		this->stride.out[0] = this->nsp * this->spatial_sz; // output batch stride
		this->stride.out[1] = this->spatial_sz; // output superpixel stride
		this->stride.in[0] = input_shape.dim_size(1) * input_shape.dim_size(2); // input batch stride
		this->stride.in[1] = input_shape.dim_size(2); // input spatial stride
		this->stride.in[2] = input_shape.dim_size(1) / output_shape.dim_size(1); // input rows / thread
		this->stride.in[3] = input_shape.dim_size(2) / output_shape.dim_size(2); // input cols / thread
	}
};

typedef SuperpixelFreqShapeBase<short> SuperpixelFreqShape;

template <typename Device, typename T_in, typename T_out>
struct SuperpixelFreqFunctor {
  void operator()(const Device& device, SuperpixelFreqShape shape, const T_in* in, T_out* out, const int test_kernel = -1);
};

}

#endif

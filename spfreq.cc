#include "spfreq.h"

#define EIGEN_USE_THREADS
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

using namespace tensorflow;

namespace spfreq {

template <typename Device, typename T_in, typename T_out>
class SuperpixelFreqOp : public OpKernel {
 public:
  explicit SuperpixelFreqOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("output_shape", &this->output_shape_attr));
    OP_REQUIRES_OK(context, context->GetAttr("test_kernel", &this->test_kernel));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);

    TensorShape output_shape(output_shape_attr);
    const int num_dims = output_shape.dims();
    OP_REQUIRES(context, num_dims == 3,
      errors::InvalidArgument("output_shape must be 3-tuple"));

    TensorShape in = input.shape();
    const int64 batch_size = in.dim_size(0);
    TensorShape out;
    out.AddDim(batch_size);
    out.AppendShape(output_shape);

    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out, &output));

    OP_REQUIRES(context, in.dim_size(1)%output_shape.dim_size(1)==0,
      errors::InvalidArgument("Dimension not evenly divisible at spatial axis(0)"));
    OP_REQUIRES(context, in.dim_size(2)%output_shape.dim_size(2)==0,
      errors::InvalidArgument("Dimension not evenly divisible at spatial axis(1)"));

    SuperpixelFreqShape spfreqShape(in, output_shape);

    SuperpixelFreqFunctor<Device, T_in, T_out>()(context->eigen_device<Device>(), spfreqShape,
      input.flat<T_in>().data(), output->flat<T_out>().data(), this->test_kernel);
  }
 private:
  TensorShape output_shape_attr;
  int test_kernel;
};

}

REGISTER_OP("SuperpixelFreq")
  .Attr("T_in: {int32}")
  .Attr("T_out: {float} = DT_FLOAT")
  .Attr("output_shape: shape")
  .Attr("test_kernel: int = -1")
  .Input("superpixels: T_in")
  .Output("spfreq_out: T_out")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle output_shape;

    ::tensorflow::shape_inference::ShapeHandle batch_size;
    TF_RETURN_IF_ERROR(c->Subshape(c->input(0), 0, 1, &batch_size));

    ::tensorflow::TensorShapeProto shape_attr;
    TF_RETURN_IF_ERROR(c->GetAttr("output_shape", &shape_attr));
    TF_RETURN_IF_ERROR(c->MakeShapeFromShapeProto(shape_attr, &output_shape));

    ::tensorflow::shape_inference::ShapeHandle out;
    TF_RETURN_IF_ERROR(c->Concatenate(batch_size, output_shape, &out));
    c->set_output(0, out);
    return Status::OK();
  });

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T_in, T_out) REGISTER_KERNEL_BUILDER( \
      Name("SuperpixelFreq").Device(DEVICE_GPU) \
      .TypeConstraint<T_in>("T_in") \
      .TypeConstraint<T_out>("T_out"), \
      spfreq::SuperpixelFreqOp<GPUDevice, T_in, T_out>);
REGISTER_GPU(int32, float);
#endif  // GOOGLE_CUDA


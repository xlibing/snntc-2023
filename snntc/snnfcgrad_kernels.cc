#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "snnfcgrad.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// CPU reference implementation of the op
// only limited function is implemented.
template <typename T>
struct SnnFcGradFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& grad_tensor, const Tensor& input_tensor, \
    const Tensor& weight_tensor, const Tensor& params_tensor, const Tensor& oldout_tensor, \
    Tensor* grad_input_tensor, Tensor* grad_weight_tensor, Tensor* grad_params_tensor, \
    const unsigned long Bi, const unsigned long Ii, const unsigned long Ji) {
    // input_tensor X[Bi, Ii] or in[Bi, Ii]. output_tensor Y[2*Bi, Ji] or oldout[2*Bi, Ji]. weight tensor W[Ii+bias, Ji]
    // true oldout: Bi*Ji.  with an extra Bi*Ji to store (\sum_w-1) for gradient calculation
    const CPUDevice& d = context->eigen_device<CPUDevice>();
    auto grad = grad_tensor.flat<T>().data();
    auto in = input_tensor.flat<T>().data();
	auto weight = weight_tensor.flat<T>().data();
	auto params = params_tensor.flat<T>().data();
	auto oldout = oldout_tensor.flat<T>().data();
    auto grad_input = grad_input_tensor->flat<T>().data();
    auto grad_weight = grad_weight_tensor->flat<T>().data();
    auto grad_params = grad_params_tensor->flat<T>().data();

    uint64 b, i, j, k, ki, kj, bi, b1i, b2j, outsize;
    uint64 size0 = Bi, size1 = Ii, size2 = Ji;
    T param0 = params[0], param4 = params[2];
    T* wsum = (T*) malloc(size0*size2*sizeof(T));

    outsize = size0 * size2;
   // set grad of params to 0 since they are not really used
   for (i = 0; i <= 4; ++i) grad_params[i] = 0.0;

   // calculate wsum: \sumw - 1.0
   for (b = 0; b < size0; ++b) {
   for (j = 0; j < size2; ++j) {
        k = b*size2+j;
        if (oldout[k] > param0 - 0.1) {wsum[k] = param0;}  // this output is not active
        else{
            wsum[k] = -param4;
            for (i = 0; i < size1; ++i) {
                if (in[b*size1+i] < oldout[k]) wsum[k] += weight[i*size2+j];
            }
        }
   }}

  // calculate input gradients dJ/dx_bi
    for (b = 0; b < size0; ++b) {
        bi = b*size1;
        for (i = 0; i < size1; ++i) {
            grad_input[bi+i] = 0.0;
            for (j = 0; j < size2; ++j) {
                k = b*size2+j;
                if ((in[bi+i]<oldout[k])&&(oldout[k]<param0-0.1))
                    grad_input[bi+i] += grad[k]*weight[i*size2+j]/wsum[k];
            }
        }
    }

  // calculate weight gradients dJ/dw_ij
   for (i = 0; i < size1; ++i) {
    for (j = 0; j < size2; ++j) {
        grad_weight[i*size2+j] = 0.0;
        for (b = 0; b < size0; ++b) {
            b1i = b * size1 + i; b2j = b * size2 +j;
            if ((in[b1i] < oldout[b2j])&&(oldout[b2j]<param0-.1))
                grad_weight[i*size2+j] += grad[b2j]*(in[b1i]-oldout[b2j])/wsum[b2j];
        }
     }
    }

  free(wsum);
}
};


// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class SnnFcGradOp : public OpKernel {
 public:
  explicit SnnFcGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& grad_tensor = context->input(0);
    const Tensor& input_tensor = context->input(1);
    const Tensor& weight_tensor = context->input(2);
    const Tensor& params_tensor = context->input(3);
    const Tensor& oldout_tensor = context->input(4);

    TensorShape grad_shape = grad_tensor.shape();
    TensorShape input_shape = input_tensor.shape();
    TensorShape weight_shape = weight_tensor.shape();
    TensorShape oldout_shape = oldout_tensor.shape();
    TensorShape params_shape = params_tensor.shape();

    const unsigned long size0 = input_shape.dim_size(0);
    const unsigned long size1 = input_shape.dim_size(1);
    const unsigned long size2 = weight_shape.dim_size(1);

    // Create gradient tensors to be output
    Tensor* grad_input_tensor = NULL;
    Tensor* grad_weight_tensor = NULL;
    Tensor* grad_params_tensor = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, weight_shape, &grad_weight_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, params_shape, &grad_params_tensor));

    OP_REQUIRES(context, size0 * 2 == grad_shape.dim_size(0),
                errors::InvalidArgument("snnfcgrad_kernels.cc: Grad/Oldout batch should be 2 * batch"));

    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("snnfcgrad_kernels.cc: Too many elements in input tensor"));
    OP_REQUIRES(context, weight_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("snnfcgrad_kernelss.cc: Too many elements in weight tensor"));
    OP_REQUIRES(context, oldout_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("snnfcgrad_kernelss.cc: Too many elements in forward output tensor"));

    SnnFcGradFunctor<Device, T>()(context, grad_tensor, input_tensor, weight_tensor, params_tensor, oldout_tensor, \
        grad_input_tensor, grad_weight_tensor, grad_params_tensor, size0, size1, size2);
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SnnFcGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SnnFcGradOp<CPUDevice, T>);

REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template struct SnnFcGradFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SnnFcGrad").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("params"), \
      SnnFcGradOp<GPUDevice, T>);


REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow

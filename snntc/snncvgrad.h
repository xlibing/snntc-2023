// head file for the c++ op of SNN-TC conv layer: backward (gradient) op
#include "tensorflow/core/framework/op_kernel.h"
#ifndef KERNEL_SNN_CV_GRAD_H_
#define KERNEL_SNN_CV_GRAD_H_
#define MAX_SPIKE_TIME 1e5

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct SnnCvGradFunctor {
  void operator()(OpKernelContext* context, const Tensor& grad_tensor, const Tensor& input_tensor, \
    const Tensor& weight_tensor, const Tensor& params_tensor, const Tensor& oldout_tensor, const int* dimparams, \
    Tensor* grad_input_tensor, Tensor* grad_weight_tensor,  Tensor* grad_params_tensor);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_SNN_CV_GRAD_H_

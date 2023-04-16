// head file for the gradient op of SNN-TC dense layer
#include "tensorflow/core/framework/op_kernel.h"
#ifndef KERNEL_SNN_FC_GRAD_H_
#define KERNEL_SNN_FC_GRAD_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct SnnFcGradFunctor {
  void operator()(OpKernelContext* context, const Tensor& grad_tensor, const Tensor& input_tensor, \
    const Tensor& weight_tensor, const Tensor& params_tensor, const Tensor& oldout_tensor, \
    Tensor* grad_input_tensor, Tensor* grad_weight_tensor,  Tensor* grad_params_tensor, \
    const unsigned long B, const unsigned long I, const unsigned long J);
};

}// namespace functor

}  // namespace tensorflow

#endif //KERNEL_SNN_FC_GRAD_H_

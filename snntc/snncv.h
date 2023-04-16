// head file for the c++ op of SNN-TC conv layer: forward op
#include "tensorflow/core/framework/op_kernel.h"
#ifndef KERNEL_SNN_CV_H_
#define KERNEL_SNN_CV_H_

namespace tensorflow {

namespace functor {
template <typename Device, typename T>
struct SnnCvFunctor {
  void operator()(OpKernelContext* context, const Tensor& input_tensor, const Tensor& weight_tensor, \
                  const Tensor& params_tensor, const int* dimparams, Tensor* output_tensor);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_SNN_CV_H_

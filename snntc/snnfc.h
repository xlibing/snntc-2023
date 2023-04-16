// head file for SNN-TC dense (fully-connected) layer ops
#include "tensorflow/core/framework/op_kernel.h"
#ifndef KERNEL_SNN_FC_H_
#define KERNEL_SNN_FC_H_

namespace tensorflow {

namespace functor {
template <typename Device, typename T>
struct SnnFcFunctor {
    void operator()(OpKernelContext* context, const Tensor& input_tensor, const Tensor& weight_tensor, \
                  const Tensor& params_tensor, Tensor* output_tensor, \
                  const unsigned long B, const unsigned long I, const unsigned long J);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_SNN_FC_H_

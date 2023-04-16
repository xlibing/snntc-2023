#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "snncvgrad.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

// CPU implementation of the backward (gradient) calculation of SNN-TC conv layer
template <typename T>
struct SnnCvGradFunctor<CPUDevice, T> {
    void operator()(OpKernelContext* context, const Tensor& grad_tensor, const Tensor& input_tensor, \
        const Tensor& weight_tensor, const Tensor& params_tensor, const Tensor& oldout_tensor, const int* dimparams, \
        Tensor* grad_input_tensor, Tensor* grad_weight_tensor, Tensor* grad_params_tensor){

    auto grad = grad_tensor.flat<T>().data();
    auto in = input_tensor.flat<T>().data();
    auto weight = weight_tensor.flat<T>().data();
	auto params = params_tensor.flat<T>().data();
	auto oldout = oldout_tensor.flat<T>().data();
    auto grad_input = grad_input_tensor->flat<T>().data();
    auto grad_weight = grad_weight_tensor->flat<T>().data();
    auto grad_params = grad_params_tensor->flat<T>().data();

    // Grad[B,Nh,Nw,Kc],in[B,H,W,C], weight[Kh*Kw*C+1, Kc], oldout[B,Nh,Nw,Kc],
    // grad_input [B,H,W,C], grad_weight[Kh*Kw*C+1,Kc], grad_params=0
    // params=[max_spike_time, Thread_per_Block, Blocks, Epsilon, Spike_Threshold, cudaflag, Kh, Kw, Sh, Sw, padding]
    // dimparams= [B, H, W, C, Kh*Kw*C+1, Kc, Nh, Nw, PadH, PadW]
    int B=dimparams[0], H=dimparams[1], W=dimparams[2], C=dimparams[3], Kh=params[6], Kw=params[7], bias=0;
    int Kc=dimparams[5], Sh=params[8], Sw=params[9], Nh=dimparams[6], Nw=dimparams[7];
    T param0=params[0], param3=params[3], param4=params[4], xbias = 1.0; //maxtime 1e5, epsilon 1e-10, threshold 1
    if (params[11] > 1e-8) {bias = 1; xbias = params[11];}

    int b, h, w, c, nh, nw, nc, kc, i, j, k, ki, kj, bi;
    int nh0, nh1, nw0, nw1, ih0, ih1, iw0, iw1, ih, iw, ic, kmax, kthreshold, kmax0;
    T currentin; T* wsum;
   // set grad of params to 0 since they are not really used
   for (i = 0; i < 10; ++i) grad_params[i] = 0.0;

   // calcualte wsum
   wsum = (T*) malloc(B*Nh*Nw*Kc*sizeof(T));
   for (b = 0; b < B; ++b) {
   for (nh = 0; nh < Nh; ++nh) {
   for (nw = 0; nw < Nw; ++nw) {
   for (kc = 0; kc < Kc; ++kc) {
      ih0 = nh*Sh; ih1 = ih0+Kh-1; iw0 = nw*Sw; iw1 = iw0+Kw-1;
      k = b*Nh*Nw*Kc + nh*Nw*Kc + nw*Kc + kc;
      if (oldout[k] < param0-.1) {
         if (bias == 1) wsum[k] = weight[Kh*Kw*C*Kc+kc]-param4;  // initialized with bias -1
         else wsum[k] = -param4;
         for (h = ih0; h <= ih1; ++h) {
         for (w = iw0; w <= iw1; ++w) {
         for (c = 0; c < C; ++c) {
              if (in[b*H*W*C+h*W*C+w*C+c] < oldout[k])
                   {wsum[k] += weight[((h-ih0)*(iw1-iw0+1)*C+(w-iw0)*C+c)*Kc+kc];}
          }}}} // end for (h, w, c)
      else wsum[k] = param0;
    }}}}  // end for (b, nh, nw, kc)

  // calculate input gradients dJ/dx_bi
    for (b = 0; b < B; ++b) {
    for (h = 0; h < H; ++h) {
    for (w = 0; w < W; ++w) {
    for (c = 0; c < C; ++c) {
        bi = b*H*W*C + h*W*C + w*C + c;
        grad_input[bi] = 0.0;
        currentin = in[bi];
        if (currentin > param0 - .1) continue;  // in=param0 means padding, just let gradient=0
        nh0 = ceil((h-Kh+1.0)/Sh); if (nh0 < 0) nh0 = 0;
        nh1 = h/Sh; if (nh1 > Nh-1) nh1 = Nh-1;
        nw0 = ceil((w-Kw+1.0)/Sw); if (nw0 < 0) nw0 = 0;
        nw1 = w/Sw; if (nw1 > Nw-1) nw1 = Nw-1;

        for (nh = nh0; nh <= nh1; ++nh) {
        for (nw = nw0; nw <= nw1; ++nw) {
            kmax = (h-nh*Sh)*Kw*C+(w-nw*Sw)*C+c;  // position of this input in this kernel block
            kmax0 = b*Nh*Nw*Kc+nh*Nw*Kc+nw*Kc;
            for (kc = 0; kc < Kc; ++kc) {
                k = kmax0+kc;
                if ((currentin<oldout[k])&&(oldout[k]<param0-.1)) { // valid input and output
                    grad_input[bi] += grad[k]*weight[kmax*Kc+kc]/wsum[k];}
             } // end for (kc)
        }} // end for (nh, nw)
    }}}}  // end for (b, h, w, c)

// ------------------------------------------------------------
// calculate weight graidents dJ/dw_ijk
  for (bi = 0; bi < Kh*Kw*C+bias; ++bi) {
  for (kc = 0; kc < Kc; ++kc) {
    grad_weight[bi*Kc+kc] = 0.0;
    if (bi == Kh*Kw*C) { // last row, bias
        for (b = 0; b < B; ++b) {
        for (nh = 0; nh < Nh; ++nh) {
        for (nw = 0; nw < Nw; ++nw) {
            kmax = b*Nh*Nw*Kc + nh*Nw*Kc + nw*Kc + kc;  // this output index
            if (oldout[kmax] < param0-.1)  // param0 means invalid output.
                grad_weight[bi*Kc+kc] += grad[kmax]*(xbias-oldout[kmax])/wsum[kmax];
        }}} continue;}  // end of bias gradient
    // other weight's gradient
    for (b = 0; b < B; ++b) {
    for (nh = 0; nh < Nh; ++nh) {
    for (nw = 0; nw < Nw; ++nw) { // for each (b, nh, nw) point, find the order rank of x corresponding to this weight bi
//        ih0 = nh*Sh; //ih1 = nh*Sh+Kh-1;
//        iw0 = nw*Sw; //iw1 = nw*Sw+Kw-1;
        h = bi/(Kw*C)+nh*Sh; w = (bi%(Kw*C))/C+nw*Sw; c = bi%C; // location of x corresponding to this weight bi
        currentin = in[b*H*W*C+h*W*C+w*C+c];  // value of this x
        if (currentin >= param0-0.1) continue;   // this is zero padding, no contribution to weight
        kmax = b*Nh*Nw*Kc + (nh*Nw+nw)*Kc + kc;  // this ouput index (b, nh, nw, kc)
           if ((currentin < oldout[kmax])&&(oldout[kmax]<param0-0.1)) // currentin is used in calculating output, so used for weight gradient
            {grad_weight[bi*Kc+kc] += grad[kmax]*(currentin-oldout[kmax])/wsum[kmax];} //oldout[2*outbatchsize+kmax];}
    }}} // end for (b, nh, nw)
   }}  //end for (bi, kc)

  free(wsum);
  }
};


// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class SnnCvGradOp : public OpKernel {
 public:
  explicit SnnCvGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
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

    auto pa = params_tensor.flat<T>().data();

    int size[10]; // B, H, W, C, Kh*Kw*C+1, Kc, Nh, Nw, PadH, PadW
    size[0] = input_shape.dim_size(0);    // B
    size[1] = input_shape.dim_size(1);    // H
    size[2] = input_shape.dim_size(2);    // W
    size[3] = input_shape.dim_size(3);    // C
    size[4] = weight_shape.dim_size(0);  // Kh*Kw*C+bias (bias =0, 1 for nobias, bias)
    size[5] = weight_shape.dim_size(1);   // Kc
    size[6] = oldout_shape.dim_size(1);   // Nh
    size[7] = oldout_shape.dim_size(2);   // Nw

    int cudaflag = pa[5], Kh = pa[6], Kw = pa[7], Sh = pa[8], Sw = pa[9], padding = pa[10], bias  = 0;
    if (pa[11] > 1e-8) bias = 1;
    if (padding == 1)   // padding == 'SAME'
        {size[8] = (ceil(size[1]/Sh)-1)*Sh-size[1]+Kh; size[9] = (ceil(size[2]/Sw)-1)*Sw-size[2]+Kw;}
    else {size[8] = 0; size[9] = 0;}
    // X already padded, we just calculate Nh,Nw with 0 padding size

    OP_REQUIRES(context, size[4] == Kh*Kw*size[3]+bias, errors::InvalidArgument("snncvgrad_kernels.cc: Input & Weight dim mismatch"));

    // Create an output tensor
    Tensor* grad_input_tensor = NULL;
    Tensor* grad_weight_tensor = NULL;
    Tensor* grad_params_tensor = NULL;

    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &grad_input_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(1, weight_shape, &grad_weight_tensor));
    OP_REQUIRES_OK(context, context->allocate_output(2, params_shape, &grad_params_tensor));

    OP_REQUIRES(context, grad_input_tensor->NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("snncvgrad_kernels.cc: Too many elements in input grad tensor"));
    OP_REQUIRES(context, grad_weight_tensor->NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("snncvgrad_kernels.cc: Too many elements in weight grad tensor"));

    SnnCvGradFunctor<Device, T>()(
        context, grad_tensor, input_tensor, weight_tensor, params_tensor, oldout_tensor, \
        size, grad_input_tensor, grad_weight_tensor, grad_params_tensor);
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SnnCvGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SnnCvGradOp<CPUDevice, T>);

REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template struct SnnCvGradFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SnnCvGrad").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("params"), \
      SnnCvGradOp<GPUDevice, T>);


REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "snncv.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

///////////////////////////////////////////////////////////////////////////////////
// MergeSort algorithm Nlog(N) complexity
template <typename T>
void CopyArray(T* B, T* A, int n, int* B1, int* A1)
{
    for(int i = 0; i < n; i++)
        {A[i] = B[i]; A1[i] = B1[i];}
}
//  Left run is A[iLeft :iRight-1].
// Right run is A[iRight:iEnd-1  ].
template <typename T>
void BottomUpMerge(T* A, int iLeft, int iRight, int iEnd, T* B, int* A1, int* B1)
{
    int i = iLeft;
    int j = iRight;
    // While there are elements in the left or right runs...
    for (int k = iLeft; k < iEnd; k++) {
        // If left run head exists and is <= existing right run head.
        if (i < iRight && (j >= iEnd || A[i] <= A[j])) {
            B[k] = A[i]; B1[k] = A1[i];
            i = i + 1;
        } else {
            B[k] = A[j]; B1[k] = A1[j];
            j = j + 1;
        }
    }
}
// array A[] has the items to sort; array B[] is a work array
template <typename T>
void BottomUpMergeSort(T* A, int n, int* A1)
{   int k1, k2;
    T* B = (T*) malloc(n*sizeof(T));
    int * B1 = (int*) malloc(n*sizeof(int));
    for (k1 = 0; k1 < n; k1++) {A1[k1] = k1; }
    // Each 1-element run in A is already "sorted".
    // Make successively longer sorted runs of length 2, 4, 8, 16... until the whole array is sorted.
    for (int width = 1; width < n; width = 2 * width)
    {
        // Array A is full of runs of length width.
        for (int i = 0; i < n; i = i + 2 * width)
        {
            // Merge two runs: A[i:i+width-1] and A[i+width:i+2*width-1] to B[]
            // or copy A[i:n-1] to B[] ( if(i+width >= n) )
            if ((i+width) < n) {k1 = i+width; } else {k1 = n;}
            if ((i+2*width) < n) {k2 = i+2*width; } else {k2 = n;}
            BottomUpMerge(A, i, k1, k2, B, A1, B1);
//            BottomUpMerge(A, i, min(i+width, n), min(i+2*width, n), B);
        }
        // Now work array B is full of runs of length 2*width.
        // Copy array B to array A for the next iteration.
        // A more efficient implementation would swap the roles of A and B.
        CopyArray(B, A, n, B1, A1);
        // Now array A is full of runs of length 2*width.
    }
    free(B1); free(B);
}
//////////////////////////////////////////////////////////////////////////////////////


// CPU implementation of the SNN-TC conv forward op
template <typename T>
struct SnnCvFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_tensor, const Tensor& weight_tensor, \
                  const Tensor& params_tensor, const int* dimparams, Tensor* output_tensor) {
    // in[B,H,W,C], weight[Kh*Kw*C+bias, Kc], out[B, Nh, Nw, Kc],
    // params=[max_spike_time, Thread_per_Block, Blocks, Epsilon, Spike_Threshold, cudaflag, Kh, Kw, Sh, Sw, padding]
    // dimparams= [B, H, W, C, Kh*Kw*C+1, Kc, Nh, Nw, PadH, PadW]

    auto in = input_tensor.flat<T>().data();
    auto weight = weight_tensor.flat<T>().data();
	auto params = params_tensor.flat<T>().data();
    auto out = output_tensor->flat<T>().data();

    int B=dimparams[0], H=dimparams[1], W=dimparams[2], C=dimparams[3], Kh=params[6], Kw=params[7], bias=0;
    int Kc=dimparams[5], Sh=params[8], Sw=params[9], Nh=dimparams[6], Nw=dimparams[7];
    T param0=params[0], param3=params[3], param4=params[4], xbias=1.0; //maxtime 1e5, epsilon 1e-10, threshold 1
    if (params[11] > 1e-8) {bias = 1; xbias = params[11];}

    int b, nh, nw, kc, gi, i, j, k, ib, wind, Ih, Iw, Ic;
    T ysum, wet, wsum, temp, prein, currentin;

    ////////////////////////////////////////////////////////////////////////////////////////
    // sort input spike time in each batch, stored to sortid[B, H*W*C].
    T* insort = (T*) malloc(H*W*C*sizeof(T));
    int* sortid = (int*) malloc(B*H*W*C*sizeof(int));  // save sorted index
    for (b = 0; b < B; ++b) {
        for (i = 0; i < H*W*C; ++i) insort[i] = in[b*H*W*C+i];  // copy to a writable array
        BottomUpMergeSort(insort, H*W*C, &sortid[b*H*W*C]);
    }
    free(insort);
    ///////////////////////////////////////////////////////////////////////////////////////////

    int* karray = (int*)malloc(Nh*Nw*Kc*sizeof(int));  // save k (# of weights used in current calculation)
    T* outB = (T*)malloc(B*Nh*Nw*Kc*sizeof(T));  // save intermediate results: original out[(B+b), Nh, Nw, Kc]
    T* outB2 = (T*)malloc(B*Nh*Nw*Kc*sizeof(T)); // save original out[(2B+b), Nh, Nw, Kc]

    // generate output spike time
    int outbatchsize = Nh*Nw*Kc, wsize = Kh*Kw*C + bias;  // a batch of output size, a batch of weight size
    int inbatchsize = H*W*C;

  for (b = 0; b < B; ++b) {
    // initialization, and bias (in=1) 1.0*weights
    for (i = 0; i < Nh*Nw; ++i) {
        for (kc = 0; kc < Kc; ++kc) {
            if (bias == 1) {wsum = weight[Kh*Kw*C*Kc+kc]-param4; ysum = xbias*weight[Kh*Kw*C*Kc+kc];}
            else {wsum = -param4; ysum = 0.0;}
            karray[i*Kc+kc] = -1e9;  // initialize k = -1e9  (bias weight only)
            if (wsum != 0) temp = ysum / wsum; else temp = ysum / param3;
            outB[b*outbatchsize+i*Kc+kc] = ysum;
            outB2[b*outbatchsize+i*Kc+kc] = wsum;
            if ((temp > xbias)&&(temp<param0))  out[b*outbatchsize+i*Kc+kc] = temp;
            else out[b*outbatchsize+i*Kc+kc] = param0*10;  // not valid spiking time
        }
    }

    for (gi = 0; gi < H*W*C; ++gi)  {
        Ih = sortid[b*inbatchsize+gi]/(W*C); Iw = (sortid[b*inbatchsize+gi]%(W*C))/C;
        currentin = in[b*inbatchsize+sortid[b*inbatchsize+gi]];  // input X[b, sortid[b, Ih, Iw, Ic]]

        int nh0 = ceil((Ih-Kh+1.0)/Sh); if (nh0 < 0) nh0 = 0;
        int nh1 = floor(((float)Ih)/Sh); if (nh1 > Nh-1) nh1 = Nh-1;
        int nw0 = ceil((Iw-Kw+1.0)/Sw); if (nw0 < 0) nw0 = 0;
        int nw1 = floor(((float)Iw)/Sw); if (nw1 > Nw-1) nw1 = Nw-1;
        for (nh = nh0; nh <= nh1; ++nh) {
            for (nw = nw0; nw <= nw1; ++nw) {
                for (kc = 0; kc < Kc; ++kc) {
                    ib = nh*(Nw*Kc) + nw * Kc + kc;  // output element position
                    k = karray[ib];   // the kth weight/input to be used
                    if (k >= 0) continue;  // this element is activated, no more input/weights allowed.
                    if (k < -1e8) {karray[ib] = 0; k = 0;}  // initial condition, only bias weight is added, set to 0

                    if (out[b*outbatchsize+ib] <= currentin)  // out[ib]>prein already satisfied, if satisfy this, then spike, no need of this weight
                        {outB[b*outbatchsize+ib] = -k; karray[ib] = -k;
                         continue;}
                    if (currentin >= param0-1.) continue;  // this is padding zero, choose to skip it (it does not spike).

                    wsum = outB2[b*outbatchsize+ib]; // read previous data
                    ysum = outB[b*outbatchsize+ib];  // read previous data

                    wind = sortid[b*inbatchsize+gi] - Ih*W*C - nw*Sw*C + (Ih-nh*Sh)*Kw*C; // this input -> weight index
                    wet = weight[wind*Kc+kc];  // get the kth weight W[k, kc]
                    wsum = wsum + wet;
                    ysum = ysum + currentin * wet;  // \sum X[b, sortid[gi]] * W[k, kc]
                    if ((wsum<param3) && (wsum>-param3)) temp = ysum/param3;  // use epsilon 1e-10 to avoid zero dividing
                    else temp = ysum / wsum;
                    // save current result. Note this may not be valid result (temp <= current)
                    if (temp > currentin) out[b*outbatchsize+ib] = temp;
                    else out[b*outbatchsize+ib] = param0*10;  // not partially good, let it be maxspiketime
                    outB[b*outbatchsize+ib] = ysum; outB2[b*outbatchsize+ib] = wsum;
                    karray[ib] -= 1;  // have considered on more input/weight, add k by 1
                } // end for (kc...)
            } // end of for nw
        }  // end of for nh
    } // end of for gi

   // at last, find those never spike, set to max spike time, and k = -1 (k=0 means bias spike)
   // if the last output > last input, then let it be spike (not set it to max spike time)
    for (i = 0; i < Nh*Nw*Kc; ++i) {
        if (karray[i] < -0.5) {
            if (out[b*outbatchsize+i] < param0)  {outB[b*outbatchsize+i] = -karray[i];}
            else
              {out[b*outbatchsize+i] = param0; outB[b*outbatchsize+i] = -1; outB2[b*outbatchsize+i] = 0.0; }
    }}
  } // end of for (b...)

  free(karray); free(outB); free(outB2); free(sortid);

  }  // end of operator() function
};  // end this function

// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class SnnCvOp : public OpKernel {
 public:
  explicit SnnCvOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const Tensor& weight_tensor = context->input(1);
    const Tensor& params_tensor = context->input(2);

    auto pa = params_tensor.flat<T>().data();

    const TensorShape& input_shape = input_tensor.shape();
    const TensorShape& weight_shape = weight_tensor.shape();
    int size[10]; // B, H, W, C, Kh*Kw*C+1, Kc, Nh, Nw, PadH, PadW
    size[0] = input_shape.dim_size(0);    // B
    size[1] = input_shape.dim_size(1);    // H
    size[2] = input_shape.dim_size(2);    // W
    size[3] = input_shape.dim_size(3);    // C
    size[4] = weight_shape.dim_size(0);  // Kh*Kw*C (no bias) or Kh*Kw*C+1 (bias)
    size[5] = weight_shape.dim_size(1);   // Kc

    int cudaflag = pa[5], Kh = pa[6], Kw = pa[7], Sh = pa[8], Sw = pa[9], padding = pa[10], bias  = 0;
    if (pa[11] > 1e-8) bias = 1;

    if (padding == 1)   // padding == 'SAME'
        {size[8] = (ceil(size[1]/Sh)-1)*Sh-size[1]+Kh; size[9] = (ceil(size[2]/Sw)-1)*Sw-size[2]+Kw;}
    else {size[8] = 0; size[9] = 0;}
    // current version: X already padded, we just calculate Nh,Nw with 0 padding size
    size[6] = (size[1]-Kh)/Sh+1; size[7] = (size[2]-Kw)/Sw+1;
    //size[6] = (size[1]-Kh+size[8])/Sh+1; size[7] = (size[2]-Kw+size[9])/Sw+1;

    OP_REQUIRES(context, size[4] == Kh*Kw*size[3]+bias, errors::InvalidArgument("snncv_kernels.cc: Input & Weight dim mismatch (bias)"));

    TensorShape output_shape;
    output_shape.AddDim(size[0] * 2);
    output_shape.AddDim(size[6]);
    output_shape.AddDim(size[7]);
    output_shape.AddDim(size[5]);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

    // check tensor size and dim
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("snncv_kernels.cc: Too many elements in input tensor"));
    OP_REQUIRES(context, weight_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("snncv_kernels.cc: Too many elements in weight tensor"));
    OP_REQUIRES(context, output_tensor->NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("snncv_kernelss.cc: Too many elements in output tensor"));
    OP_REQUIRES(context, size[0] * size[6] * size[7] * size[4] <= tensorflow::kint32max,
                errors::InvalidArgument("snncv_kernels.cc: Too many elements in patched tensor"));

    SnnCvFunctor<Device, T>()(
        context, input_tensor, weight_tensor, params_tensor, size, output_tensor);
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SnnCv").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SnnCvOp<CPUDevice, T>);

REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.  // added HostMemory to keep params in CPU so we can access it
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template struct SnnCvFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SnnCv").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("params"), \
      SnnCvOp<GPUDevice, T>);


REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow

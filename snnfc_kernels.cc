#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif

#include "snnfc.h"
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


// CPU version of the SNN_dense forward op.
template <typename T>
struct SnnFcFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_tensor, const Tensor& weight_tensor, \
      const Tensor& params_tensor, Tensor* output_tensor, \
      const unsigned long B, const unsigned long I, const unsigned long J) {
    // input_tensor X[B, I] or in[B, I]. output_tensor Y[2*B, J] or out[2*B, J]. weight tensor W[I+bias, J]
    // true out: B*J.  with an extra B*J to store (\sum_w-1) for gradient calculation
    uint64 b, i, j, ki, kj, outsize;
    uint64 size0 = B, size1 = I, size2 = J;
    T ysum, w, wsum, temp;

    auto in = input_tensor.flat<T>().data();
    auto weight = weight_tensor.flat<T>().data();
	auto params = params_tensor.flat<T>().data();
    auto out = output_tensor->flat<T>().data();

    outsize = size0*size2;

////////////////////////////////////////////////////////////////////////////////////////
// sort input spike time in each batch, stored to insort
    T* insort = (T*) malloc(size0*size1*sizeof(T));
    int* sortid = (int*) malloc(size0*size1*sizeof(int));  // save sorted index

    for (i = 0; i < size0*size1; ++i) {
        insort[i] = in[i];
    }
    for (b = 0; b < size0; ++b) {
        BottomUpMergeSort(&insort[b*size1], size1, &sortid[b*size1]);
    }
///////////////////////////////////////////////////////////////////////////////////////////

    // generate output spike time
    for (i = 0; i < outsize; ++i) {
        out[i] = params[0];  //MAX_SPIKE_TIME;
        out[outsize + i] = 0.0;
    }

    for (b = 0; b < size0; ++b) {
        ki = b * size1;  // start position of this input batch
        kj = b * size2;  // start position of this output batch
        for (j = 0; j < size2; ++j) {
            wsum = -params[2]; ysum = 0.0;  // save (\sum w - 1.0)
            if (params[5] > 1e-8)  { // there is bias
                w = weight[size1*size2+j]; wsum += w; ysum += params[5] * w;
                if ((wsum>params[1]) || (wsum<-params[1]))
                    {temp = ysum / wsum;
                     if ((temp>params[5]) && (temp <= in[ki+sortid[ki]]))
                         {out[kj+j] = temp; out[outsize+kj+j] = wsum; break;}
                     }
            }

            for (i = 0; i < size1; ++i) {
                w = weight[sortid[ki+i]*size2+j]; //weight W[sortid[ki+i], j]
                wsum = wsum + w;
                ysum = ysum + in[ki+sortid[ki+i]]*w; //   insort[ki+i]*w;  // in X[b, sortid[ki+i]]
                if ((wsum<params[1]) && (wsum>-params[1])) continue;  // params[3]=1e-10, epsilon, avoid zero dividing
                temp = ysum/wsum;
               // if (wsum<params[3]) temp = ysum/params[3];  // tf ref version using this clipping
                if (temp>in[ki+sortid[ki+i]]) {
                    if (i < size1-1)
                        {if (temp<=in[ki+sortid[ki+i+1]])
                           {out[kj+j] = temp; out[outsize + kj + j] = wsum;
                            break; }
                           }
                    else   {if (temp<params[0])  //  (ysum/(wsum-1.)<MAX_SPIKE_TIME)
                              {out[kj+j] = temp; out[outsize + kj + j] = wsum;
                               }
                              }
                }
            }
        }
    }

    free(sortid); free(insort);

  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors, but we use float32 only.
template <typename Device, typename T>
class SnnFcOp : public OpKernel {
 public:
  explicit SnnFcOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    const Tensor& weight_tensor = context->input(1);
    const Tensor& params_tensor = context->input(2);

    const TensorShape& input_shape = input_tensor.shape();
    const TensorShape& weight_shape = weight_tensor.shape();
    const unsigned long size0 = input_shape.dim_size(0);
    const unsigned long size1 = input_shape.dim_size(1);
    const unsigned long size2 = weight_shape.dim_size(1);
    // if no bias, then size1 is one less than rows of weight
    OP_REQUIRES(context, (weight_shape.dim_size(0) == size1) || (weight_shape.dim_size(0) == size1+1),
                errors::InvalidArgument("snnfc_kernels.cc: Input & Weight inner-dim mismatch"));

    TensorShape output_shape;
    output_shape.AddDim(size0 * 2);
    output_shape.AddDim(size2);

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output_tensor));

    // check default sizes in case not sure:
    //    printf("kint32max = %d\n", tensorflow::kint32max);  // int32_t or int  // 2**31 or 2GB
    //    printf("int bytes = %ld\n", sizeof(int));   // sizeof: long unsigned int  // 4 bytes
    //    printf("unsigned int bytes = %ld\n", sizeof(unsigned int));   // 4 bytes
    //    printf("unsigned long bytes = %ld\n", sizeof(unsigned long));  // 8 bytes
    //    printf("long unsigned int bytes = %ld\n", sizeof(long unsigned int));  // 8 bytes
    //    printf("unsigned long long byptes = %ld\n", sizeof(unsigned long long));  // 8 bytes
    //    printf("uint64 byptes = %ld\n", sizeof(uint64));  // 8 bytes
    //    printf("size_t bytes = %ld\n", sizeof(size_t));   // 8 bytes
    //    printf("uint32 bytes = %ld\n", sizeof(uint32));   // 4 bytes

    // Check tensor size & dim
    OP_REQUIRES(context, input_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("snnfc_kernels.cc: Too many elements in input tensor"));
    OP_REQUIRES(context, size0 * size1 <= tensorflow::kint32max,
                errors::InvalidArgument("snnfc_kernels.cc: Too many elements in input tensor"));
    OP_REQUIRES(context, weight_tensor.NumElements() <= tensorflow::kint32max,
                errors::InvalidArgument("snnfc_kernels.cc: Too many elements in weight tensor"));
    OP_REQUIRES(context, size1 * size2 <= tensorflow::kint32max,
                errors::InvalidArgument("snnfc_kernels.cc: Too many elements in weight tensor"));
    OP_REQUIRES(context, size0 * size2 * 2 <= tensorflow::kint32max,
                errors::InvalidArgument("snnfc_kernels.cc: Too many elements in output tensor"));

    SnnFcFunctor<Device, T>()(
        context, input_tensor, weight_tensor, params_tensor, output_tensor, size0, size1, size2);
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SnnFc").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      SnnFcOp<CPUDevice, T>);

REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template struct SnnFcFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("SnnFc").Device(DEVICE_GPU).TypeConstraint<T>("T").HostMemory("params"), SnnFcOp<GPUDevice, T>);


REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow

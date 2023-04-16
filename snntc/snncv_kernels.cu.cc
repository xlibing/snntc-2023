#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "snncv.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cub/cub.cuh>
using namespace cub;

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// GPU kernel of calculate output spike times
// if biased case, one more row in weight [II+1, JJ]
// bias corresponding to input value biasvolt, which can be arbitrary. It is processed first.
// But negative biasvolt still needs careful thought. So far we assume biasvolt >=0, & biasvolt >1e-8 means bias.
template <typename T>
__global__ void SnnFcCudaOutKernelBias(const T* in, const T* weight, const int* sortid, T* out, T* out2, \
  const int BB, const int II, const int JJ, const T params0, const T params1, const T params2, const T biasvolt) {
  //  params0, params1, params2: maxspiketime 1e5, epsilon 1e-10, spike threshold 1.0
  // in(BB, II), weight(II+bias, JJ), out(BB, JJ), sortid(BB, II)

  unsigned int k, B, J, ord, Bii;
  bool threadstatus = false;
  T accu = params0, wsum = -params2, ysum = 0.0, wsum0=0.0, x, w, param0comp = params0 - 0.1, wsumabs, ysumabs;

  B = blockIdx.y * blockDim.y + threadIdx.y; // output location (B, J) for this thread
  J = blockIdx.x * blockDim.x + threadIdx.x; //
  Bii = B * II;

  if ((B < BB) && (J < JJ)) {   // no need check B < BB because it is always correct. we check it here for sure.
    // take care of biasvolt*bias first (i.e., x=biasvolt; k=-1)
    if (biasvolt > 1e-8) {
        w = weight[II * JJ + J]; wsum += w; ysum = biasvolt * w;
        wsumabs = wsum; ysumabs = ysum;
        if (wsumabs < 0) {wsumabs = -wsumabs; ysumabs = -ysumabs;}
        threadstatus = (ysumabs > wsumabs * biasvolt);
    }

    // take care of all other in * weight for small in to large in
    for (k = 0; k < II; k++) {
        x = in[Bii + k];
        if (x > param0comp) break;
        if (threadstatus) {    //if (threadstatus && (accu <= x))  break;
            if (ysumabs <= x * wsumabs) break;
        }
        ord = sortid[Bii + k];
        w = weight[ord * JJ + J];
        wsum += w;
        ysum += x * w;
        wsumabs = wsum; ysumabs = ysum;
        if (wsumabs < 0) {wsumabs = -wsumabs; ysumabs = -ysumabs;}
        threadstatus = (ysumabs > x * wsumabs);

        // the above code actually implements the following without division.
        //        if ((wsum < params1) && (wsum > -params1)) wsum = params1;
        //        wsum0 = __fdividef(1.0, wsum);
        //        accu = ysum * wsum0;
        //        threadstatus = (accu > x);
        }

    if (threadstatus) {   //if (!threadstatus) {accu = params0; wsum0 = 0.0;}
        wsum0 = __fdividef(1.0, wsum); accu = ysum * wsum0;
        if (accu >= params0) {accu = params0; wsum0 = 0.0;}
    }

    k = B * JJ + J;
    out[k] = accu; out2[k] = wsum0;
  }
}


// subroutines used by cub sort
struct SegmentOffsetCreator {
  EIGEN_DEVICE_FUNC SegmentOffsetCreator(int num_cols) : num_cols_(num_cols) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(int idx) const {
    return idx * num_cols_;
  };
  int num_cols_;
};

struct ColumnIndexCreator {
  ColumnIndexCreator(int num_cols) : num_cols_(num_cols) {}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int operator()(
      const Eigen::array<int, 1>& ix) const {
    return ix[0] % num_cols_;
  }
  int num_cols_;
};


// GPU implementation of the SNN-TC conv forward op
template <typename T>
struct SnnCvFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_tensor, const Tensor& weight_tensor, \
                  const Tensor& params_tensor, const int* dimparams, Tensor* output_tensor) {
    const T* weight = weight_tensor.flat<T>().data();
	const T* params = params_tensor.flat<T>().data();
    T* out = output_tensor->flat<T>().data();
    const GPUDevice& d = context->eigen_device<GPUDevice>();

    // in[B,H,W,C], weight[Kh*Kw*C+bias, Kc], out[B, Nh, Nw, Kc],
    // params=[max_spike_time, Thread_per_Block, Blocks, Epsilon, Spike_Threshold, cudaflag, Kh, Kw, Sh, Sw, padding, biasvolt]
    // dimparams= [B, H, W, C, Kh*Kw*C+bias, Kc, Nh, Nw, PadH, PadW]
    int B=dimparams[0], H=dimparams[1], W=dimparams[2], C=dimparams[3], Kh=params[6], Kw=params[7];
    int Kc=dimparams[5], Sh=params[8], Sw=params[9], Nh=dimparams[6], Nw=dimparams[7];
    int Bi = B * Nh * Nw, Ii = Kh * Kw * C;

/************** image patches (B*Nh*Nw, Kh*Kw*C) ********************/
    Tensor image_patch_tensor;
    TensorShape BIshape;
    BIshape.AddDim(Bi); BIshape.AddDim(Ii);
    auto err = context->allocate_temp(DT_FLOAT, BIshape, &image_patch_tensor);
    if (!err.ok()) printf("Error: snncv_kernel.cu.cc allocate CUDA memory to image_patch_tensor: %d\n", err.code());
    auto image_patch_tensor_eigen = To32Bit(image_patch_tensor.shaped<T, 4>({B, Nh, Nw, Ii})); //         tensor<T, 4>()); //.shaped<T, 4>(); //({B, Nh, Nw, Kh*Kw*C});
    auto input_tensor_eigen = To32Bit(input_tensor.shaped<T, 4>({B, H, W, C})); //           tensor<T, 4>()); // shaped<T, 4>; // ({B, H, W, C});
    image_patch_tensor_eigen.device(d) = input_tensor_eigen.extract_image_patches(\
        Kw, Kh, Sw, Sh, 1, 1, Eigen::PADDING_VALID).reshape(image_patch_tensor_eigen.dimensions());

/*********************** sort ***********************/
    auto in = image_patch_tensor.flat<T>().data();
    Tensor d_in_tensor, order_tensor;
    err = context->allocate_temp(DT_FLOAT, BIshape, &d_in_tensor);
    if (! err.ok())
        printf("Error: snncv_kernel.cu.cc allocate CUDA memory to d_in_tensor: %d\n", err.code());
    err = context->allocate_temp(DT_INT32, BIshape, &order_tensor);
    if (! err.ok())
        printf("Error: snncv_kernel.cu.cc allocate CUDA memory to order_tensor: %d\n", err.code());
    T *d_in = d_in_tensor.flat<T>().data();
    int *order = order_tensor.flat<int>().data();

    cub::CountingInputIterator<int> counting_iter(0);
    cub::TransformInputIterator<int, SegmentOffsetCreator, cub::CountingInputIterator<int>>
        d_offsets(counting_iter, SegmentOffsetCreator(Ii));

    // Create temporary tensor: column index, used for sortpairs
    Tensor input_indices_tensor;
    err = context->allocate_temp(DT_INT32, TensorShape({Bi, Ii}), &input_indices_tensor);
    if (! err.ok())
        printf("Error: snncv_kernel.cu.cc allocate CUDA memory to input_indices_tensor: %d\n", err.code());
    auto input_indices_t = To32Bit(input_indices_tensor.flat<int>());
    input_indices_t.device(d) = input_indices_t.generate(ColumnIndexCreator(Ii));
    int* input_indices = input_indices_tensor.flat<int>().data();

    // Allocate temporary storage
    size_t  temp_storage_bytes  = 0;  //    void    *d_temp_storage     = NULL;

    // older cuda supports radixsort only. if so, turn the following on:
    //    auto err1 = cub::DeviceSegmentedRadixSort::SortPairs(nullptr, temp_storage_bytes, in, d_in, \
    //            input_indices, order, Bi * Ii, Bi, d_offsets, d_offsets+1, 0, sizeof(T)*8, d.stream());
    auto err1 = cub::DeviceSegmentedSort::SortPairs(nullptr, temp_storage_bytes, in, d_in, \
            input_indices, order, Bi * Ii, Bi, d_offsets, d_offsets + 1, d.stream());
    if (err1 != cudaSuccess) printf("Error: snncv_kernel.cu.cc DeviceSegmentedSort1 %s\n", cudaGetErrorString(err1));

    Tensor d_temp_storage;
    err = context->allocate_temp(DT_INT8, TensorShape({static_cast<int64>(temp_storage_bytes)}), &d_temp_storage);
    if (! err.ok())
        printf("Error: snncv_kernel.cu.cc allocate CUDA memory to d_temp_storage: %d\n", err.code());

    //    err1 = cub::DeviceSegmentedRadixSort::SortPairs(d_temp_storage.flat<int8>().data(), temp_storage_bytes, in, d_in, \
    //            input_indices, order, Bi * Ii, Bi, d_offsets, d_offsets+1, 0, sizeof(T)*8, d.stream());
    err1 = cub::DeviceSegmentedSort::SortPairs(d_temp_storage.flat<int8>().data(), temp_storage_bytes, in, d_in, \
            input_indices, order, Bi * Ii, Bi, d_offsets, d_offsets + 1, d.stream());
    if (err1 != cudaSuccess) printf("Error: snncv_kernel.cu.cc DeviceSegmentedSort2 %s\n", cudaGetErrorString(err1));
/*********************** end of sort *****************************/

    /** output tensors are: in (Bi*Ii, original), d_in (Bi*Ii) ordered data, order (Bi*Ii) index **/

    // calculate output (d_in * weight).
    int thread_per_block = params[1], max_Y_block = params[2];

    if (thread_per_block <= 0) thread_per_block = 256;
    if (max_Y_block <= 0) max_Y_block = 16384;  //32768;

    // each row of in tensor is treated individually due to its own sort order.
    // tile & shared memory are used for each row of in and the entire weight tensor
    // Y-dim block is always 1
    // Number of Blocks(X,Y): (Ji/thread_per_block, Bi)
    // Number of threads per block(X,Y): (thread_per_block, 1)

    int Ji = Kc;
    int i = Ji / 32; i *= 32; if (i < Ji) i += 32;
    if (i < thread_per_block) thread_per_block = i;
    dim3 dimBlock(thread_per_block, 1);  // Block size (threads in x_dim, threads in y_dim)

    int x = Ji / thread_per_block;  // total number of blocks in x_dim (max is 2^31)
    if (x * thread_per_block < Ji) x++;
    int y = Bi;   // total number of blocks in y_dim
    if (y < max_Y_block) max_Y_block = y;
    dim3 dimGrid(x, max_Y_block);

    // in our scheme, Bi can be extremely large, larger than 65536
    // so we calculate in*weight in slices of in
    int kend;
    for (int k = 0; k < y; k+=max_Y_block) {
            kend = k + max_Y_block;
            if (kend > y) kend = y;
            SnnFcCudaOutKernelBias<T><<<dimGrid, dimBlock, 0, d.stream()>>>\
                    (&d_in[k*Ii], weight, &order[k*Ii], &out[k*Ji], &out[(Bi+k)*Ji], \
                    kend - k, Ii, Ji, params[0], params[3], params[4], params[11]);
            cudaError_t cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess)
                printf("cuda kernel failed in snncv_kernels.cu.cc (k=%d, kend=%d) with error \"%s\".\n", \
                k, kend, cudaGetErrorString(cudaerr));
    }

  }
};

// Explicitly instantiate functors for the types of OpKernels registered.

template struct SnnCvFunctor<GPUDevice, float>;
template struct SnnCvFunctor<GPUDevice, int32>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA

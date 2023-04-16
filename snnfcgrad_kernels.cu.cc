#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "snnfcgrad.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

extern __shared__ char shared_memory[];

template <typename T>  // gradient of the input X (dJ/dX)
__global__ void SnnFcGradCudaKernelInput(const T* grad, const T* in, const T* weight,
  const T* oldout, T* out, const T param0, const int size0, const int size1, const int size2) {
  // param0 = 1e5, param1 = 1e-10, param2 = 1 (threshold)
  // grad(size0, size2), in(size0, size1), weight(size1+bias, size2), oldout(2*size0, size2). out(size0, size1)
  // Atile: Grad; Btile: in; Ctile: oldout
  T* shared_memory_T = (T*)shared_memory;
  unsigned int i = blockDim.x * blockDim.y;
  T *Atile = &shared_memory_T[0], *Btile = &shared_memory_T[i], *Ctile = &shared_memory_T[i * 2];

  unsigned int tileDimX = blockDim.x, Tiles;  // tiles are 16x16 or 32x32
  unsigned int BB = size0, II = size1, JJ = size2;  // in(BB, II), oldout(BB, JJ), grad(BB, JJ), out(BB, II)
  unsigned int j, k, B, I;
  T accu = 0.0, xv, yv, bv, av, param0comp = param0 - 0.1;

  B = blockIdx.y * blockDim.y + threadIdx.y; // output location (B, I) for this thread
  I = blockIdx.x * blockDim.x + threadIdx.x; //

  if ((B < BB) && (I < II)) xv = in[B * II + I]; // in[B, I]
  Tiles = (JJ + tileDimX - 1) / tileDimX;  // total number of tiles

  unsigned int tilei = threadIdx.y * tileDimX + threadIdx.x; // this subroutine use all the same index
  unsigned int tilej = threadIdx.x * tileDimX + threadIdx.y;  // for weight Btile to avoid bank conflict
  unsigned int xy = blockIdx.x * blockDim.x + threadIdx.y;

  for (unsigned int tileIdx=0; tileIdx<Tiles; tileIdx++){
    // load one tile of Grad, one tile of weight, one tile of oldout into shared memory
    yv = param0; bv = param0;  // use yv, bv instead of Ctile, Btile
    j = tileIdx * tileDimX + threadIdx.x;      // column of Grad[B, j]
    if ((B < BB) && (j < JJ)) {
        k = B * JJ + j;
        Atile[tilei] = __ldg(&grad[k]); // Grad[B, j]  // av = __ldg(&grad[k]);
        yv = __ldg(&oldout[k]);  // Ctile[tilei] = __ldg(&oldout[k]); // oldout[B, j]
    }
    if ((xy < II) && (j < JJ)) {
        bv = __ldg(&weight[xy * JJ + j]); // Btile[tilej] = __ldg(&weight[xy * JJ + j]);  // weight[I, j]
    }
    Btile[tilej] = bv; Ctile[tilei] = yv;
    __syncthreads();

    // update one tile of out from tiles of Grad and weight in shared memory
    if ((B < BB) && (I < II)) {      //    #pragma unroll
        j = threadIdx.y * tileDimX; i = threadIdx.x;
        for (k = 0; k < tileDimX; k++){
            yv = Ctile[j];
            if ((xv > yv) || (yv > param0comp)) {j++; i += tileDimX; continue;}
            av = Atile[j];  // if (av > param0comp) continue;
            bv = Btile[i];  // bv = Btile[k * tileDimX + threadIdx.x]; //if (bv > param0comp) continue;
            i += tileDimX; j++;
            accu +=  av * bv;
    }}
    __syncthreads();
  }

 // save results of this thread to out[B, I] which is grad dJ/dx_bi
  if ((B < BB) && (I < II)) {out[B * II + I] = accu;}
}


template <typename T>   // gradient of weights dJ/dw
__global__ void SnnFcGradCudaKernelWeight(const T* grad, const T* in, const T* oldout, T* out,
  const T param0, const T biasvolt, const int size0, const int size1, const int size2) {
  // param0 = 1e5, param1 = 1e-10, param2 = 1 (time threshold)
  // Atile: In or X_bi;  Btile: Oldout or Y_bj;  Ctile: Grad/Wsum _bj
  T* shared_memory_T = (T*)shared_memory;
  unsigned int i = blockDim.x * blockDim.y;
  T *Atile = &shared_memory_T[0], *Btile = &shared_memory_T[i], *Ctile = &shared_memory_T[i*2];

  unsigned int tileDimX = blockDim.x, tileDimY = blockDim.y, Tiles;  // tiles are 16x16 or 32x32
  unsigned int BB = size0, II = size1, JJ = size2;  // in(BB, II), oldout(BB, JJ), grad(BB, JJ), out(II, JJ)
  unsigned int b, k, j, I, J, IIbias;
  T accu = 0.0, av, bv, cv, param0comp=param0-0.1;
  if (biasvolt > 1e-8) IIbias = II + 1; else IIbias = II;

  I = blockIdx.y * blockDim.y + threadIdx.y; // output location (I, J) for this thread
  J = blockIdx.x * blockDim.x + threadIdx.x; // location (threadIdx.y, threadIdx.x) inside this block for this thread
  Tiles = (BB + tileDimX - 1) / tileDimX;  // total number of tiles

  j = threadIdx.y * tileDimX + threadIdx.x;   // index of this shared memory location
  for (unsigned int tileIdx=0; tileIdx<Tiles; tileIdx++){
    // load one tile of in, one tile of oldout, one tile of Grad into shared memory
    // original in saved to [B, I] --> change to [I, B] in Atile
    av = param0; bv = param0;  // use av, bv to repsace Atile[j], Btile[j] temporarily
    b = tileIdx * tileDimY + threadIdx.y;      // input row index, in[b, I] is read by this thread
    k = blockIdx.y * blockDim.y + threadIdx.x; // element index of in
    if (b < BB) {
        if (k < II) {av = __ldg(&in[b * II + k]); } //   Atile[j] = __ldg(&in[b * II + k]);
        else if ((k == II) && (biasvolt > 1e-8)) av = biasvolt; // {Atile[j] = biasvolt; } //x value corresponding to bias
    }
    k = b * JJ + J;     // element index of oldout and grad
    if ((b < BB) && (J < JJ)) {
        bv = __ldg(&oldout[k]);     // oldout[b, J] is read by this thread
        Ctile[j] = __ldg(&grad[k]); // * __ldg(&oldout[BB * JJ + k]);
    }
    Atile[j] = av; Btile[j] = bv;
    __syncthreads();

    // update one tile of out from tiles of in and weight in shared memory
    if ((I < IIbias) && (J < JJ)) {  //  #pragma unroll
        b = threadIdx.x;  i = threadIdx.y;
        for (k = 0; k < tileDimX; k++){
            av = Atile[i]; bv = Btile[b];
            b += tileDimX; i += tileDimX;
            if ((av > param0comp) || (bv > param0comp) || (av > bv)) continue;
            cv = Ctile[b - tileDimX];
            accu += (av - bv) * cv;
            // ref: the above code does the following:
            //            av = Atile[k * tileDimX + threadIdx.y];
            //            bv = Btile[k * tileDimX + threadIdx.x];
            //            cv = Ctile[k * tileDimX + threadIdx.x];
            //            if ((av > param0comp) || (bv > param0comp) || (cv > param0comp)) continue;
            //            if ((av > bv) || (bv > param0comp)) continue;
    }}
    __syncthreads();
  }

 // save results of this thread to out[I, J] which is grad dJ/dw_ij
  if ((I < IIbias) && (J < JJ)) {out[I * JJ + J] = accu; }
}


// GPU implementation of the backward SNN-TC dense op
template <typename T>
struct SnnFcGradFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& grad_tensor, const Tensor& input_tensor, \
    const Tensor& weight_tensor, const Tensor& params_tensor, const Tensor& oldout_tensor, \
    Tensor* grad_input_tensor, Tensor* grad_weight_tensor, Tensor* grad_params_tensor, \
    const unsigned long Bi, const unsigned long Ii, const unsigned long Ji) {

    auto d = context->eigen_device<GPUDevice>();
    auto in = input_tensor.flat<T>().data();
	auto weight = weight_tensor.flat<T>().data();
	auto params = params_tensor.flat<T>().data();
	auto oldout = oldout_tensor.flat<T>().data();
    auto grad_input = grad_input_tensor->flat<T>().data();
    auto grad_weight = grad_weight_tensor->flat<T>().data();
    auto grad_params = grad_params_tensor->flat<T>().data();

    int bias = 0;
    if (params[5] > 1e-8) bias = 1;

    // calculate grad/wsum using eigen tensor
    // grad_tensor[2*Bi,Ji], oldout_tensor[2*Bi, Ji] --> grad_tensor[0:Bi, 0:Ji] * oldout_tensor[Bi:2*Bi, 0:Ji]
    // oldout_tensor.slice({Bi,0}, {2*Bi, Ji})  // Eigen::array<Eigen::Index, 2> offsets = {Bi, 0}
    Tensor up_grad_tensor;
    TensorShape BJshape;
    BJshape.AddDim(Bi); BJshape.AddDim(Ji);
    auto err = context->allocate_temp(DT_FLOAT, BJshape, &up_grad_tensor);
    if (! err.ok()) printf("Error: snnfcgrad_kernel.cu.cc allocate CUDA memory to up_grad_tensor: %d\n", err.code());
    auto up_grad_tensor_eigen = To32Bit(up_grad_tensor.shaped<T, 2>({(long)Bi, (long)Ji}));   // {B, Nh, Nw, Ii})); //         tensor<T, 4>()); //.shaped<T, 4>(); //({B, Nh, Nw, Kh*Kw*C});
    auto grad_tensor_eigen = To32Bit(grad_tensor.shaped<T, 2>({(long)(2*Bi), (long)Ji})); // ({B, H, W, C})); //           tensor<T, 4>()); // shaped<T, 4>; // ({B, H, W, C});
    auto oldout_tensor_eigen = To32Bit(oldout_tensor.shaped<T, 2>({(long)(2*Bi), (long)Ji}));
    Eigen::array<Eigen::Index, 2> offsets0 = {0, 0};
    Eigen::array<Eigen::Index, 2> extents0 = {(long)Bi, (long)Ji};
    Eigen::array<Eigen::Index, 2> offsets = {(long)Bi, 0};
    Eigen::array<Eigen::Index, 2> extents = {long(Bi), (long)Ji};
    up_grad_tensor_eigen.device(d) = grad_tensor_eigen.slice(offsets0, extents0) \
                                    * oldout_tensor_eigen.slice(offsets, extents);
    auto up_grad = up_grad_tensor.flat<T>().data();

    // calculate grad of input
    int thread_per_block = params[3], max_Y_block = params[4];
    if (thread_per_block <= 0) thread_per_block = 256;
    if (max_Y_block <= 0) max_Y_block = 16384;  //32768;

    uint64 tilethreads = 16;
    if (thread_per_block == 1024) tilethreads = 32;
    if (thread_per_block == 64) tilethreads = 8;
    dim3 dimBlock(tilethreads, tilethreads);   // threads_per_block is the size of tile

    uint64 x = Ii / tilethreads;
    if (x * tilethreads < Ii) x++;
    uint64 y = Bi / tilethreads;
    if (y * tilethreads < Bi) y++;
    if (y < max_Y_block) max_Y_block = y; // we can reduce block dim when data is small
    dim3 dimGrid((int)x, max_Y_block);   // Y-dim max block is limited to 65536, X-dim is 2^31

    // in our scheme, Bi can be extremely large, larger than 65536
    // so we calculate in slices of in & grad_input
    uint64 kend;
    cudaError_t cudaerr;
    for (uint64 k = 0; k < Bi; k+=max_Y_block*tilethreads) {
            kend = k + max_Y_block * tilethreads;
            if (kend > Bi) kend = Bi;
            SnnFcGradCudaKernelInput<T><<<dimGrid, dimBlock, 3*thread_per_block*sizeof(T), d.stream()>>>\
                (&up_grad[k*Ji], &in[k*Ii], weight, &oldout[k*Ji], \
                 &grad_input[k*Ii], params[0], (int)(kend-k), (int)Ii, (int)Ji);
            cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess)
                printf("*** cuda kernel failed in snnFcGradCudaKernelInput (k=%ld, kend=%ld) with error \"%s\".\n", \
                k, kend, cudaGetErrorString(cudaerr));
    }

    // calculate grad of weight
    x = Ji / tilethreads;
    if (x * tilethreads < Ji) x++;
    y = (Ii + bias) / tilethreads;
    if (y * tilethreads < (Ii + bias)) y++;
    dim3 dimGrid2(x, y);   // y < 65536, or Ii < 65536*16=2^20
    if (y > 65535)
       printf("*** cuda error: snnfcgrad_kernels.cu.cc snnFcGradCudaKernelWeight Y-dim block = %ld > 65535\n", y);

    SnnFcGradCudaKernelWeight<T>
        <<<dimGrid2, dimBlock, 3*thread_per_block*sizeof(T), d.stream()>>>\
            (up_grad, in, oldout, grad_weight, params[0], params[5], (int)Bi, (int)Ii, (int)Ji);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("*** cuda kernel failed in snnFcGradCudaKernelWeight (Bi=%ld, Ii=%ld, Ji=%ld) with error \"%s\".\n", \
        Bi, Ii, Ji, cudaGetErrorString(cudaerr));
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.

template struct SnnFcGradFunctor<GPUDevice, float>;
template struct SnnFcGradFunctor<GPUDevice, int32>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA

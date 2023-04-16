#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "snncvgrad.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

extern __shared__ char shared_memory[];

template <typename T>  // gradient to inputs  dJ/dX
__global__ void SnnCvGradCudaKernelInput(const T* grad, const T* in, const T* weight,
  const T* oldout, T* out, const T param0, const int BS, const int BB,
  const int Ba, const int H, const int W, const int C, const int Nh, const int Nw, const int JJ, \
  const int Kh, const int Kw, const int Sh, const int Sw) {
  // BS:start BB of this block. BB: total BB of this block
  // param0 = 1e5
  // Atile: Grad; Btile: in; Ctile: oudout
  T* shared_memory_T = (T*)shared_memory;
  unsigned int i = blockDim.x * blockDim.y;
  T *Atile = &shared_memory_T[0], *Btile = &shared_memory_T[i], *Ctile = &shared_memory_T[i * 2];

  unsigned int tileDimX = blockDim.x, Tiles;  // tiles are 16x16 or 32x32
  // in(BB, II), oldout(BB, JJ), grad(BB, JJ), out(BB, II)
  // BS: start BB of this block, used for converting to original in(B, H, W, C)
  unsigned int j, k, B, I, II = Kh * Kw * C;
  unsigned int a, nh, nw, kh, kw, c, h, w, ori;
  T accu = 0.0, xv, yv, av, bv, param0comp = param0 - 0.1;

  B = blockIdx.y * blockDim.y + threadIdx.y; // output location (B, I) for this thread
  I = blockIdx.x * blockDim.x + threadIdx.x; //

  // no patches in[BB, II]. we need to convert it to original in[B, H, W, C]
  if ((B < BB) && (I < II)) { //xv = in[B * II + I]; // in[B, I] --> should really be in[BS+B, I]
        a = (BS + B) / (Nh * Nw); nh = ((BS + B) % (Nh * Nw)) / Nw; nw = (BS + B) % Nw;
        kh = I / (Kw * C); kw = (I % (Kw * C)) / C; c = I % C;
        h = nh * Sh + kh; w = nw * Sw + kw;
        ori = a * H * W * C + h * W * C + w * C + c;
        xv = __ldg(&in[ori]);
  }
  Tiles = (JJ + tileDimX - 1) / tileDimX;  // total number of tiles

  unsigned int tilei = threadIdx.y * tileDimX + threadIdx.x; // this subroutine use all the same index
  unsigned int tilej = threadIdx.x * tileDimX + threadIdx.y;  // for weight Btile to avoid bank conflict
  unsigned int xy = blockIdx.x * blockDim.x + threadIdx.y;

  for (unsigned int tileIdx=0; tileIdx<Tiles; tileIdx++){
    // load one tile of Grad, one tile of weight, one tile of oldout into shared memory
    yv = param0; bv = param0; // use yv, bv instead of Ctile, Btile
    j = tileIdx * tileDimX + threadIdx.x;      // column of Grad[B, j]
    if ((B < BB) && (j < JJ)) {
        k = B * JJ + j;
        Atile[tilei] = __ldg(&grad[k]); // * __ldg(&oldout2[k]);     // Grad[B, j]
        yv = __ldg(&oldout[k]);  // oldout[B, j]
    }
    if ((xy < II) && (j < JJ)) {
        bv = __ldg(&weight[xy * JJ + j]);  // weight[I, j]
    }
    Btile[tilej] = bv; Ctile[tilei] = yv;
    __syncthreads();

    // update one tile of out from tiles of Grad and weight in shared memory
    if ((B < BB) && (I < II)) {      //    #pragma unroll
        j = threadIdx.y * tileDimX; i = threadIdx.x;
        for (k = 0; k < tileDimX; k++){
            yv = Ctile[j];
            if ((xv > yv) || (yv > param0comp)) {j++; i += tileDimX; continue;}
            av = Atile[j]; //if (av > param0comp) continue;
            bv = Btile[i];  // bv = Btile[k * tileDimX + threadIdx.x]; //if (bv > param0comp) continue;
            i += tileDimX; j++;
            accu +=  av * bv;
    }}
    __syncthreads();
  }

 // save results of this thread to out[BS+B, I] which is grad dJ/dx_bi
 // patched out[BS, I] not available, we add them to original out[B, H, W, C] or out[a, h, w, c] or out[ori]
  if ((B < BB) && (I < II)) //{out[B * II + I] = accu;}  //{out[ori] = accu; } //0.0;
    {atomicAdd(&out[ori], accu);}
}


// CUDA kernel of calculating weight gradient  dJ/dw
// when there is bias, weight[II+1, JJ], input is biasvolt
// There is no in(BB, II), but just original input in(B, H, W, C). Need to convert in(b, k) to in (b, h, w, c)
template <typename T>
__global__ void SnnCvGradCudaKernelWeightBias(const T* grad, const T* in, const T* oldout, T* out, const T param0, \
  const T biasvolt, const int B, const int H, const int W, const int C, const int Nh, const int Nw, const int JJ, \
  const int Kh, const int Kw, const int Sh, const int Sw) {
  // param0 = 1e5
  // Atile: In or X_bi;  Btile: Oldout or Y_bj;  Ctile: Grad/Wsum _bj
  T* shared_memory_T = (T*)shared_memory;
  unsigned int i = blockDim.x * blockDim.y;
  T *Atile = &shared_memory_T[0], *Btile = &shared_memory_T[i], *Ctile = &shared_memory_T[i*2];

  unsigned int tileDimX = blockDim.x, tileDimY = blockDim.y, Tiles;  // tiles are 16x16 or 32x32
  unsigned int BB = B * Nh * Nw, II = Kh * Kw * C, IIbias; // in(BB, II), oldout(BB, JJ), grad(BB, JJ), out(II, JJ)
  unsigned int b, k, j, I, J;
  unsigned int a, nh, nw, kh, kw, c, h, w;
  T accu = 0.0, av, bv, cv, param0comp=param0-0.1;
  if (biasvolt > 1e-8) IIbias = II + 1; else IIbias = II;

  I = blockIdx.y * blockDim.y + threadIdx.y; // output location (I, J) for this thread
  J = blockIdx.x * blockDim.x + threadIdx.x; // location (threadIdx.y, threadIdx.x) inside this block for this thread
  Tiles = (BB + tileDimX - 1) / tileDimX;  // total number of tiles

  j = threadIdx.y * tileDimX + threadIdx.x;   // index of this shared memory location
  for (unsigned int tileIdx=0; tileIdx<Tiles; tileIdx++){
    // load one tile of in, one tile of oldout, one tile of Grad into shared memory
    // original in saved to [B, I] --> change to [I, B] in Atile
    av = param0; bv = param0; // use av, bv as Atile, Btile temporarily
    b = tileIdx * tileDimY + threadIdx.y;      // input row index, in[b, I] is read by this thread
    k = blockIdx.y * blockDim.y + threadIdx.x; // element index of in
    if (b < BB) {
        if (k < II) {
            a = b / (Nh * Nw); nh = (b % (Nh * Nw)) / Nw; nw = b % Nw;
            kh = k / (Kw * C); kw = (k % (Kw * C)) / C; c = k % C;
            h = nh * Sh + kh; w = nw * Sw + kw;
            av = __ldg(&in[a * H * W * C + h * W * C + w * C + c]);
        }
        else if ((k == II) && (biasvolt > 1e-8)) // x value corresponding to bias
            av = biasvolt;
    }

    k = b * JJ + J;     // element index of oldout and grad
    if ((b < BB) && (J < JJ)) { // oldout[b, J] is read by this thread
        bv = __ldg(&oldout[k]);
        Ctile[j] = __ldg(&grad[k]);
    }
    Atile[j] = av; Btile[j] = bv;
    __syncthreads();

    // update one tile of out from tiles of in and weight in shared memory
    if ((I < IIbias) && (J < JJ)) {  //  #pragma unroll
        b = threadIdx.x; i = threadIdx.y;
        for (k = 0; k < tileDimX; k++){
            av = Atile[i]; bv = Btile[b];
            b += tileDimX; i += tileDimX;
            if ((av > param0comp) || (bv > param0comp) || (av > bv)) continue;
            cv = Ctile[b - tileDimX];
            accu += (av - bv) * cv;

            // the above codes this following:
            //            av = Atile[k * tileDimX + threadIdx.y];
            //            bv = Btile[k * tileDimX + threadIdx.x];
            //            cv = Ctile[k * tileDimX + threadIdx.x];
                //            if ((av > param0comp) || (bv > param0comp) || (cv > param0comp)) continue;
                //            if ((av > bv) || (bv > param0comp)) continue;
            //            if ((av > param0comp) || (bv > param0comp) || (av > bv)) continue;
            //            accu += (av - bv) * cv;
    }}
    __syncthreads();
  }

 // save results of this thread to out[I, J] which is grad dJ/dw_ij
  if ((I < IIbias) && (J < JJ)) {out[I * JJ + J] = accu; }
}

// GPU implementation of the backward (gradient) op of SNN-TC conv layer
template <typename T>
struct SnnCvGradFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& grad_tensor, const Tensor& input_tensor, \
        const Tensor& weight_tensor, const Tensor& params_tensor, const Tensor& oldout_tensor, const int* dimparams, \
        Tensor* grad_input_tensor, Tensor* grad_weight_tensor, Tensor* grad_params_tensor){

    auto d = context->eigen_device<GPUDevice>();
    auto in = input_tensor.flat<T>().data();
    auto weight = weight_tensor.flat<T>().data();
	auto params = params_tensor.flat<T>().data();
	auto oldout = oldout_tensor.flat<T>().data();
    auto grad_input = grad_input_tensor->flat<T>().data();
    auto grad_weight = grad_weight_tensor->flat<T>().data();
    auto grad_params = grad_params_tensor->flat<T>().data();

    // Grad[2*B,Nh,Nw,Kc],in[B,H,W,C], weight[Kh*Kw*C+bias, Kc], oldout[2*B,Nh,Nw,Kc],
    // grad_input [B,H,W,C], grad_weight[Kh*Kw*C+bias,Kc], grad_params=0
    // params=[max_spike_time, Thread_per_Block, Blocks, Epsilon, Spike_Threshold, cudaflag, Kh, Kw, Sh, Sw, padding, bias]
    // dimparams= [B, H, W, C, Kh*Kw*C+bias, Kc, Nh, Nw, PadH, PadW]
    int B=dimparams[0], H=dimparams[1], W=dimparams[2], C=dimparams[3], Kh=params[6], Kw=params[7], bias=0;
    int Kc=dimparams[5], Sh=params[8], Sw=params[9], Nh=dimparams[6], Nw=dimparams[7];
    int Bi = B * Nh * Nw, Ii = Kh * Kw * C, Ji = Kc;
    if (params[11] > 1e-8) bias = 1;

    // calculate grad/wsum using eigen tensor
    // grad_tensor[2B, Nh, Nw, Kc], oldout_tensor[2B, Nh, Nw, Kc] --> grad_tensor[:B,:,:,:] * oldout_tensor[B:,:,:,:]
    Tensor up_grad_tensor;
    TensorShape BJshape;
    BJshape.AddDim(B); BJshape.AddDim(Nh); BJshape.AddDim(Nw); BJshape.AddDim(Kc);
    auto err = context->allocate_temp(DT_FLOAT, BJshape, &up_grad_tensor);
    if (! err.ok()) printf("Error: snncvgrad_kernel.cu.cc allocate CUDA memory to up_grad_tensor: %d\n", err.code());
    auto up_grad_tensor_eigen = To32Bit(up_grad_tensor.shaped<T, 4>({B, Nh, Nw, Kc})); //         tensor<T, 4>()); //.shaped<T, 4>(); //({B, Nh, Nw, Kh*Kw*C});
    auto grad_tensor_eigen = To32Bit(grad_tensor.shaped<T, 4>({2*B, Nh, Nw, Kc})); //           tensor<T, 4>()); // shaped<T, 4>; // ({B, H, W, C});
    auto oldout_tensor_eigen = To32Bit(oldout_tensor.shaped<T, 4>({2*B, Nh, Nw, Kc}));
    Eigen::array<Eigen::Index, 4> offsets0 = {0, 0, 0, 0};
    Eigen::array<Eigen::Index, 4> extents0 = {B, Nh, Nw, Kc};
    Eigen::array<Eigen::Index, 4> offsets = {B, 0, 0, 0};
    Eigen::array<Eigen::Index, 4> extents = {B, Nh, Nw, Kc};
    up_grad_tensor_eigen.device(d) = grad_tensor_eigen.slice(offsets0, extents0) \
                                    * oldout_tensor_eigen.slice(offsets, extents);
    auto up_grad = up_grad_tensor.flat<T>().data();

    // initialize input grad to 0 because we will add multiple grads to it
    auto grad_input_tensor_eigen = grad_input_tensor->flat<T>();
    grad_input_tensor_eigen.device(d) = grad_input_tensor_eigen.constant(T(0));

    // start cuda grad calculation
    int thread_per_block = params[1], max_Y_block = params[2];
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
    cudaError_t cudaerr;

    uint64 kend;
    for (uint64 k = 0; k < Bi; k+=max_Y_block*tilethreads) {
            kend = k + max_Y_block * tilethreads;
            if (kend > Bi) kend = Bi;
            SnnCvGradCudaKernelInput<T><<<dimGrid, dimBlock, 3*thread_per_block*sizeof(T), d.stream()>>>\
                (&up_grad[k*Ji], in, weight, &oldout[k*Ji], \
                 grad_input, params[0], (int)k, (int)(kend-k), \
                 B, H, W, C, Nh, Nw, Kc, Kh, Kw, Sh, Sw);
            cudaerr = cudaDeviceSynchronize();
            if (cudaerr != cudaSuccess)
                printf("*** cuda kernel failed in snnCvGradCudaKernelInput (k=%ld, kend=%ld) with error \"%s\".\n", \
                k, kend, cudaGetErrorString(cudaerr));
    }

    // calculate grad of weight
    x = Ji / tilethreads;
    if (x * tilethreads < Ji) x++;
    y = (Ii + bias) / tilethreads;
    if (y * tilethreads < (Ii + bias)) y++;
    dim3 dimGrid2(x, y);   // y < 65536, or Ii < 65536*16=2^20
    if (y > 65535)
       printf("*** cuda error: snncvgrad_kernels.cu.cc snnCvGradCudaKernelWeight Y-dim block = %ld > 65535\n", y);

    SnnCvGradCudaKernelWeightBias<T>
            <<<dimGrid2, dimBlock, 3*thread_per_block*sizeof(T), d.stream()>>>\
                (up_grad, in, oldout, grad_weight, params[0], params[11], B, H, W, C, Nh, Nw, Kc, Kh, Kw, Sh, Sw);
    cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("*** cuda kernel failed in snnCvGradCudaKernelWeight (Bi=%d, Ii=%d, Ji=%d) with error \"%s\".\n", \
        Bi, Ii, Ji, cudaGetErrorString(cudaerr));

  }
};

// Explicitly instantiate functors for the types of OpKernels registered.

template struct SnnCvGradFunctor<GPUDevice, float>;
template struct SnnCvGradFunctor<GPUDevice, int32>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA

#!/usr/bin/bash

TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )

TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )

nvcc --compiler-options -std=c++14 -c -o snncv_kernels.cu.o snncv_kernels.cu.cc ${TF_CFLAGS[@]} \
-D_GLIBCXX_USE_CXX11_ABI=1 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr \
-Wno-deprecated-gpu-targets --ptxas-options=-v
#\
#-gencode=arch=compute_86,code=sm_86

g++ -std=c++14 -shared -o snncv_ops.so snncv_kernels.cc snncv.h snncv_ops.cc snncv_kernels.cu.o \
${TF_CFLAGS[@]} -D_GLIBCXX_USE_CXX11_ABI=1 -D GOOGLE_CUDA=1  -fPIC -O3 \
 ${TF_LFLAGS[@]} -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart

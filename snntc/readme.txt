----------------------- Introduction --------------------------------
SNN-TC (spiking neural network with temporal coding) has special 
forward and backward expressions that use SORT operation instead of 
conventional nonlinear activations. It is very inefficient to
implement them in python.
Even a simple CIFAR10 model would exhaust GPU memory.

This package provides c++/cuda implementation of SNN-TC dense and
and 2-D convolutional layers for the Tensorflow platform. With this,
large deep models can be implemented without too much GPU memory.

The speed is hurdled mainly by the SORT operation. The current 
version use the same amount of GPU memory as conventional
CNN but runs at 5 times slower.

We provide mainly the dense and 2D convolutional layers as c++/cuda
coding. Other layers like pooling are also provided,
which are implemented by calling python Tensorflow modules directly.
One can add his own layers via either slight modification of 
Tensorflow or following our c++/cuda programming style.

Last updated on April 15, 2023.

Developed under python 3.9, Tensorflow 2.9, cuda 11.6, ubuntu 20.4.
Tested (and optimized) mainly on a workstation with 4 Nvidia A5000
GPUs and another computer with a Nvidia 2080ti GPU.

The package includes both GPU and CPU operation source code. 
But the CPU version is very limited, e.g. it works for the case 
without bias only.
This is because this package aims for training large models
where GPU is necessary and CPU-only modules are not useful.

-------------------- Usage -----------------------------------------
Use 
     "from SNN_TC_Modules import SNN_conv, SNN_dense" 
or just
     "import SNN_TC_Modules" 
in your python code, and use the SNN modules in the same way as CNN. 

The file "SNN_TC_Modules.py" can be copied to your project folder.
There is no need to move any of the rest files. Instead, just change
"C_MODULE_ROOT = './'" to the correct directory where the 4 c++ ops 
modules *.so are in.

------------------ C++ Compilation ----------------------------------
All the *.so modules need to be recompiled when moving to a new computer.
Run 
       bash make-snnfc.sh
       bash make-snnfcgrad.sh
       bash make-snncv.sh
       bash make-snncvgrad.sh
to get the four c++ ops modules: snnfc_ops.so, snnfcgrad_ops.so,
snncv_ops.so, snncvgrad_ops.so.

During compilation, the compiler may report "missing *h file error".
Usually this is due to path mismatch. You just need to change some
c++ head/include files, such as gpu_device-functions.h
& gpu_kernel_helper.h in python-tensorflow directory.
It is easy to find these files from the error report.
Just find these *.h files, and make necessary changes, e.g.,
making the following changes:
    #include "cuComplex.h"  //"third_party/gpus/cuda/include/cuComplex.h"
    #include "cuda.h" //"third_party/gpus/cuda/include/cuda.h"
    #include "cuda_fp16.h" //"third_party/gpus/cuda/include/cuda_fp16.h"
where "third_party ..." was removed
because this may not be the true directory.

Another issue is that some older version of CUDA support RadixSort only.
If so, one has to change from segmentedsort to segmentedRadioxSort
in 2 *.cu files.

-------------------- Testing Examples ----------------------------------
There is a list of testing files that compare c++ based modules with
the pure python modules, and calculate their difference.
These files are used when debugging the c++ source codes:

    test-snn-fc-forward.py,
    test-snn-fc-backward.py: check the forward/backward of a dense layer
    test-snn-fc-layers.py,
    test-snn-fc-layers-nobias.py: test on two dense layers

    test-snn-cv-forward.py,
    test-snn-cv-backward.py: check the forward/backward of a 2D conv layer
    test-snn-cv-layers.py: test on two conv layers

These testing files also show how to use the c++ module,
either in class wrapper or in bare function.

------------------ File History --------------------------------------
From oldest to newest:
# snnconv3-snnfc10: use tf+py to do padding & sorting before calling c++ ops
# snnconv4-snnfc11: use tf+py for padding, C++ for sorting, using less memory
# snnconv5-snnfc12: use c++ for both padding and sorting, too slow

# snnconv6fc13:
#           since all of the above are too slow, we use TF as much as possible
#           just the critical part such as forward/backward is coded in c++
#           in copy2, we use tiles and shared memory in order to speed up,
#           but can not deal with large dim or tensor size

# snnconv7fc14:
#           Since all of the above are still too slow, we use cub sort
#           & tf cuda-memory allocation & eigen tensor param passing to speedup

# snnconv8fc15:
#           More GPU/tensor programming techniques are applied to enhance speed
#           Reduce python-tf pre-processing as much as possible. Specifically,
#           we use cub sort, tf memory allocation, eigen c++ patch extraction,
#           GPU shared memory, etc.
#           We also tried using sgemm (snnconv8fc15copy3) but it is not faster.
#           The reason might be
#               1) SNN-TC foward/backward ops need sort, which is dominantly slow,
#                  other techniques have minor advantages only
#               2) SNN-TC ops needs too many resources like registers and
#                  shared-memory, so sgemm leads to low GPU utilization.

speed comparison (train one epoch of GoogleNet over ImageNet,
                  using 4 A5000 GPUs in tfmodels framework):
    snnconv6fc13: 3 hour 20 min per epoch, 388 sec test on tfmodels,
                  c++ contraction only, tf sort
    snnconv7fc14old: 1 hour 50 min per epoch, 148 sec test on tfmodels,
                  c++ contraction only, cub sort
    snnconv7fc14: 1 hour 45 min per epoch, 138 sec test on tfmodels,
                  cub sort, eigen tensor param passing
    snnconv8fc15v1:
       1 hour 39 min per epoch, 115 sec test on tfmodels, eigen c++ patch extraction
       1 hour 28 min per epoch, 100 sec test, eigen c++ im2col, eigen mul for grad
    snnconv8fc15v2: 1 hour 50 min per epoch, 104 sec test,
                    same above but use x-dim as height
    snnconv8fc15copy3: save as v1 but using sgemm to calculate weight gradient.

The current version "SNN_TC_Modules" is a copy of "snnconv8fc15v3"
which is similar to snnconv8fc15v1 but with biasvolt set flexibly (not just 1.0).








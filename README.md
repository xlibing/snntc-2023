# snntc-2023
spiking neural network with temporal coding: c++/cuda &amp; tensorflow ops

This is the improved soruce code with c++ modules for the paper: 
S. Zhou, X. Li, Y. Chen, S. Chandrasekaran, and A. Sanyal, 
"Temporal-coded deep spiking neural network with easy training and robust performance," 
the Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21), Feb. 2021.
Please cite this paper if you find it this code useful.

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

Developed under python 3.9, Tensorflow 2.9, cuda 11.6, ubuntu 20.4.
Tested (and optimized) mainly on a workstation with 4 Nvidia A5000
GPUs and another computer with a Nvidia 2080ti GPU.

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

Last updated: April 16, 2023, by Xiaohua Li

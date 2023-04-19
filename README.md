# Spiking Neural Network with Temporal Coding: c++/cuda &amp; tensorflow ops

This package provides improved source code with c++/cuda modules for the paper: 

- *S. Zhou, X. Li, Y. Chen, S. Chandrasekaran, and A. Sanyal, "Temporal-coded deep spiking neural network with easy training and robust performance," the Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21), Feb. 2021.*

Please cite this paper if you find this package useful.

*SNN-TC (spiking neural network with temporal coding)* has special
forward and backward expressions that use SORT operation instead of
conventional nonlinear activations. The major challenge is that it is very inefficient to
implement them with just python and tensorflow, in terms of both speed and GPU memory. 
Even a simple CIFAR10 model would exhaust GPU memory if implemented by Tensorflow/Pytorch functions.

This package provides c++/cuda implementation of SNN-TC dense and
2-D convolutional layers for the Tensorflow platform. With this,
large deep models can be implemented and trained more efficiently.

Models constructed with the SNN-TC modules in general use the same amount of GPU memory as conventional
CNN but run at 5 times slower. It seems that the relatively slow speed is mainly due to the SORT operation. 
We have tried to implement dense and 2D conv ops with more efficient GEMM or more fancy GPU programming techniques, but they do not improve speed.

We provide the dense and 2D convolutional layers as c++/cuda
modules, whose source codes are snnfc*.* and snncv*.*, respectively. 
Other layers like pooling are also provided, but are implemented
in SNN_TC_Modules.py by calling python Tensorflow functions directly.
One can add his own layers via either slight modification of
Tensorflow or following our c++/cuda programming style.

Developed under python 3.9, Tensorflow 2.9, cuda 11.6, ubuntu 20.4.
Tested (and optimized) mainly on a workstation with 4 Nvidia A5000
GPUs and another computer with a Nvidia 2080ti GPU (cuda 11.0).

## Usage 

Use

     import SNN_TC_Modules as SNN 
     
or

     from SNN_TC_Modules import SNN_conv, SNN_dense
     
in your python code, and use the SNN modules in the same way as CNN.
See snntc-mnist.py, snntc-cifar10.py, and snntc-imagenet.py for usage examples.

The file "SNN_TC_Modules.py" can be copied to your project folder.
There is no need to move any of the rest files, but you need to change
"C_MODULE_ROOT = './'" to the correct directory where the 4 c++ ops
modules *.so are in.

## C++ Compilation 

All the *.so modules need to be recompiled when moving to a new computer.
Run

       bash make-snnfc.sh
       
       bash make-snnfcgrad.sh
       
       bash make-snncv.sh
       
       bash make-snncvgrad.sh
       
to get the four c++ ops modules: snnfc_ops.so, snnfcgrad_ops.so,
snncv_ops.so, snncvgrad_ops.so.

---

*Last updated: April 16, 2023, by Xiaohua Li*

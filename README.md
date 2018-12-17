# reference-neural-network

This C++ library contains a library for training and running simple feed-forward fully-connected neural networks. Network structure is specified at compile time through a list of template parameters. An example usage can be found in the included "example.cpp" file which trains a small network to fit four input-output pairs using stochastic gradient descent.

# Basic Usage
```C++
auto network = neural::corpus<double, 4, 3, 1>(); //creates an empty network with 2 layers (4 -> 3 and 3 -> 1)
```
Additional SGD optimization code can be found in "example.cpp"

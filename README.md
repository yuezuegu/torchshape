# Torchshape
Calculates the output shape of Pytorch operations

How to use:

1) Install torchshape package:

    ``pip install torchshape``

2) Import torchshape package:

    ``from torchshape import tensorshape``
    
3) Call tensorshape function in your code:

    ``outshape = tensorshape(op, inshape)``

Parameters:

- op (torch.nn.Container) - Single or collection of operations such as nn.Module, nn.Sequential, or nn.ModuleList
- inshape (tuple of integers) - Dimensions of expected input tensor. First element is always batch size and second element is number of input channels. For image-based tensors, third and forth dimensions are image height and image width.

See [lenet.py](lenet.py) for example usage.

## List of supported operations
- nn.Conv1d
- nn.Conv2d
- nn.Linear
- nn.MaxPool1d
- nn.MaxPool2d
- nn.AvgPool1d
- nn.AvgPool2d
- nn.Flatten
- nn.BatchNorm1d
- nn.BatchNorm2d

# Torchshape
Calculates the output shape of Pytorch operations

How to use:

1) Install torchshape package:

    ``pip install torchshape``

2) Import torchshape package:

    ``from torchshape import tensorshape``
    
3) Call tensorshape function in your code:

    ``outshape = tensorshape(op, inshape)``

where op is a torch.nn operation (see the [list](#list-of-supported-operations) of supported operations), inshape and outshape are tuples.

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

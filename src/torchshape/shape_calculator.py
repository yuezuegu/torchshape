
import torch
import torch.nn as nn

import math 

def tensorshape_(op, in_shape):
    if isinstance(op, nn.Conv2d):
        return tensorshape_conv(op, in_shape)
    elif isinstance(op, nn.BatchNorm2d):
        return in_shape
    elif isinstance(op, nn.ReLU):
        return in_shape
    elif isinstance(op, nn.MaxPool2d) or isinstance(op, nn.AvgPool2d):
        return tensorshape_pooling(op, in_shape)
    else:
        Warning ("Operation of type {} is not supported, returning in_shape".format(op.__class__.__name__))
        return in_shape 

def tensorshape_conv(op, in_shape):
    N, Cin, Hin, Win = in_shape

    Cout = op.out_channels

    kernel_size = (op.kernel_size, op.kernel_size) if isinstance(op.kernel_size, int) else op.kernel_size
    padding = (op.padding, op.padding) if isinstance(op.padding, int) else op.padding
    dilation = (op.dilation, op.dilation) if isinstance(op.dilation, int) else op.dilation
    stride = (op.stride, op.stride) if isinstance(op.stride, int) else op.stride

    groups = op.groups

    if isinstance(padding, str):
        # https://pytorch.org/docs/master/_modules/torch/nn/modules/conv.html#Conv2d
        # self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
        # if padding == 'same':
        #     for d, k, i in zip(dilation, kernel_size,
        #                         range(len(kernel_size) - 1, -1, -1)):
        #         total_padding = d * (k - 1)
        #         left_pad = total_padding // 2
        #         self._reversed_padding_repeated_twice[2 * i] = left_pad
        #         self._reversed_padding_repeated_twice[2 * i + 1] = (
        #             total_padding - left_pad)
        raise NotImplementedError ("Padding=same or valid is not implemented")

    assert op.in_channels % groups == 0, "Cin must be a multiple of groups"

    if groups == 1:
        assert Cin == groups * op.in_channels, "Input channels must be the same: C_weight: {}, C_input: {}".format(op.in_channels, Cin)

    Hout = math.floor( (Hin + 2*padding[0] - dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1 )
    Wout = math.floor( (Win + 2*padding[1] - dilation[1] * (kernel_size[1]-1) - 1) / stride[1] + 1 )

    return (N, Cout, Hout, Wout)

def tensorshape_pooling(op, in_shape):
    N, Cin, Hin, Win = in_shape

    kernel_size = (op.kernel_size, op.kernel_size) if isinstance(op.kernel_size, int) else op.kernel_size
    padding = (op.padding, op.padding) if isinstance(op.padding, int) else op.padding

    if hasattr(op, 'dilation'):
        dilation = (op.dilation, op.dilation) if isinstance(op.dilation, int) else op.dilation  
    else: 
        dilation = (1,1)

    stride = (op.stride, op.stride) if isinstance(op.stride, int) else op.stride

    if isinstance(padding, str):
        # https://pytorch.org/docs/master/_modules/torch/nn/modules/conv.html#Conv2d
        # self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
        # if padding == 'same':
        #     for d, k, i in zip(dilation, kernel_size,
        #                         range(len(kernel_size) - 1, -1, -1)):
        #         total_padding = d * (k - 1)
        #         left_pad = total_padding // 2
        #         self._reversed_padding_repeated_twice[2 * i] = left_pad
        #         self._reversed_padding_repeated_twice[2 * i + 1] = (
        #             total_padding - left_pad)
        raise NotImplementedError ("Padding=same or valid is not implemented")

    if op.ceil_mode:
        Hout = math.ceil( (Hin + 2*padding[0] - dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1 )
        Wout = math.ceil( (Win + 2*padding[1] - dilation[1] * (kernel_size[1]-1) - 1) / stride[1] + 1 )
    else:
        Hout = math.floor( (Hin + 2*padding[0] - dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1 )
        Wout = math.floor( (Win + 2*padding[1] - dilation[1] * (kernel_size[1]-1) - 1) / stride[1] + 1 )

    return (N, Cin, Hout, Wout)

def tensorshape(op, in_shape):
    assert hasattr(op, "_modules"), "Op object is expected to have _modules attribute"

    if len(op._modules) > 0:
        for k,m in op._modules.items():
            in_shape = tensorshape(m, in_shape)
    else:
        in_shape = tensorshape_(op, in_shape)

    return in_shape


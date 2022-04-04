
import warnings
import torch
import torch.nn as nn
import unittest

import math 

def tensorshape_(op, in_shape):
    if isinstance(op, nn.Conv1d):
        return tensorshape_conv1d(op, in_shape)
    elif isinstance(op, nn.Conv2d):
        return tensorshape_conv2d(op, in_shape)
    elif isinstance(op, nn.Linear):
        return tensorshape_linear(op, in_shape)
    elif isinstance(op, (nn.BatchNorm1d, nn.BatchNorm2d)):
        return in_shape
    elif isinstance(op, nn.ReLU, nn.Sigmoid, nn.Tanh):
        return in_shape
    elif isinstance(op, (nn.MaxPool1d, nn.AvgPool1d)):
        return tensorshape_pooling1d(op, in_shape)
    elif isinstance(op, (nn.MaxPool2d, nn.AvgPool2d)):
        return tensorshape_pooling2d(op, in_shape)
    elif isinstance(op, nn.Flatten):
        return tensorshape_flatten(op, in_shape)
    else:
        warnings.warn("Operation of type {} is not supported, returning in_shape".format(op.__class__.__name__))
        return in_shape

def tensorshape_flatten(op, in_shape):
    start_dim = op.start_dim
    end_dim = op.end_dim

    out_shape = list(in_shape[0:start_dim])

    tot = 1
    for s in in_shape[start_dim:end_dim]:
        tot = tot * s
    tot = tot * in_shape[end_dim]
    out_shape += [tot]

    if end_dim != -1:
        out_shape += in_shape[end_dim+1:]

    return tuple(out_shape)

def tensorshape_linear(op, in_shape):
    N, Cin = in_shape

    Cout = op.out_features

    assert Cin == op.in_features, "Input channels must be the same: C_weight: {}, C_input: {}".format(op.in_channels, Cin)

    return (N, Cout)

def tensorshape_conv1d(op, in_shape):
    N, Cin, Hin = in_shape

    Cout = op.out_channels

    kernel_size = (op.kernel_size,) if isinstance(op.kernel_size, int) else op.kernel_size
    
    dilation = (op.dilation,) if isinstance(op.dilation, int) else op.dilation
    stride = (op.stride,) if isinstance(op.stride, int) else op.stride

    groups = op.groups

    assert op.in_channels % groups == 0, "Cin must be a multiple of groups"

    if groups == 1:
        assert Cin == groups * op.in_channels, "Input channels must be the same: C_weight: {}, C_input: {}".format(op.in_channels, Cin)

    if isinstance(op.padding, str):
        if op.padding == "valid":
            Hout = math.floor( (Hin + dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1 )
        elif op.padding == "same":
            Hout = Hin
    else:
        padding = (op.padding,) if isinstance(op.padding, int) else op.padding

        Hout = math.floor( (Hin + 2*padding[0] - dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1 )

    return (N, Cout, Hout)

def tensorshape_conv2d(op, in_shape):
    N, Cin, Hin, Win = in_shape

    Cout = op.out_channels

    kernel_size = (op.kernel_size, op.kernel_size) if isinstance(op.kernel_size, int) else op.kernel_size
    
    dilation = (op.dilation, op.dilation) if isinstance(op.dilation, int) else op.dilation
    stride = (op.stride, op.stride) if isinstance(op.stride, int) else op.stride

    groups = op.groups

    assert op.in_channels % groups == 0, "Cin must be a multiple of groups"

    if groups == 1:
        assert Cin == groups * op.in_channels, "Input channels must be the same: C_weight: {}, C_input: {}".format(op.in_channels, Cin)

    if isinstance(op.padding, str):
        if op.padding == "valid":
            Hout = math.floor( (Hin + dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1 )
            Wout = math.floor( (Win + dilation[1] * (kernel_size[1]-1) - 1) / stride[1] + 1 )
        elif op.padding == "same":
            Hout = Hin
            Wout = Win
    else:
        padding = (op.padding, op.padding) if isinstance(op.padding, int) else op.padding

        Hout = math.floor( (Hin + 2*padding[0] - dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1 )
        Wout = math.floor( (Win + 2*padding[1] - dilation[1] * (kernel_size[1]-1) - 1) / stride[1] + 1 )

    return (N, Cout, Hout, Wout)

def tensorshape_pooling1d(op, in_shape):
    N, Cin, Hin = in_shape

    kernel_size = (op.kernel_size,) if isinstance(op.kernel_size, int) else op.kernel_size
    padding = (op.padding,) if isinstance(op.padding, int) else op.padding

    if hasattr(op, 'dilation'):
        dilation = (op.dilation,) if isinstance(op.dilation, int) else op.dilation  
    else: 
        dilation = (1,)

    stride = (op.stride,) if isinstance(op.stride, int) else op.stride

    if op.ceil_mode:
        round_op = math.ceil
    else:
        round_op = math.floor

    if isinstance(op.padding, str):
        if op.padding == "valid":
            Hout = round_op( (Hin + dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1 )
        elif op.padding == "same":
            Hout = Hin
    else:
        padding = (op.padding,) if isinstance(op.padding, int) else op.padding

        Hout = round_op( (Hin + 2*padding[0] - dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1 )

    return (N, Cin, Hout)

def tensorshape_pooling2d(op, in_shape):
    N, Cin, Hin, Win = in_shape

    kernel_size = (op.kernel_size, op.kernel_size) if isinstance(op.kernel_size, int) else op.kernel_size
    padding = (op.padding, op.padding) if isinstance(op.padding, int) else op.padding

    if hasattr(op, 'dilation'):
        dilation = (op.dilation, op.dilation) if isinstance(op.dilation, int) else op.dilation  
    else: 
        dilation = (1,1)

    stride = (op.stride, op.stride) if isinstance(op.stride, int) else op.stride

    if op.ceil_mode:
        round_op = math.ceil
    else:
        round_op = math.floor

    if isinstance(op.padding, str):
        if op.padding == "valid":
            Hout = round_op( (Hin + dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1 )
            Wout = round_op( (Win + dilation[1] * (kernel_size[1]-1) - 1) / stride[1] + 1 )
        elif op.padding == "same":
            Hout = Hin
            Wout = Win
    else:
        padding = (op.padding, op.padding) if isinstance(op.padding, int) else op.padding

        Hout = round_op( (Hin + 2*padding[0] - dilation[0] * (kernel_size[0]-1) - 1) / stride[0] + 1 )
        Wout = round_op( (Win + 2*padding[1] - dilation[1] * (kernel_size[1]-1) - 1) / stride[1] + 1 )

    return (N, Cin, Hout, Wout)

def tensorshape(op, in_shape):
    assert hasattr(op, "_modules"), "Op object is expected to have _modules attribute"

    if len(op._modules) > 0:
        for k,m in op._modules.items():
            in_shape = tensorshape(m, in_shape)
    else:
        in_shape = tensorshape_(op, in_shape)

    return in_shape



class Test(unittest.TestCase):
    def test_conv1d(self):
        x = torch.rand(size=(32,100,300))
        op = nn.Conv1d(in_channels=100, out_channels=200, kernel_size=(3), stride=(2), padding=(5), dilation=(3), groups=2)
        y = op(x)
        self.assertEqual(y.shape, tensorshape(op, x.shape))

    def test_conv2d(self):
        x = torch.rand(size=(32,100,224,224))
        op = nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(3,5), stride=(2,1), padding=(5,1), dilation=(3,2), groups=4)
        y = op(x)
        self.assertEqual(y.shape, tensorshape(op, x.shape))

    def test_maxpool1d(self):
        x = torch.rand(size=(32,100,224))
        op = nn.MaxPool1d(kernel_size=(4), stride=(3), padding=(2), dilation=(2))
        y = op(x)
        self.assertEqual(y.shape, tensorshape(op, x.shape))

    def test_maxpool2d(self):
        x = torch.rand(size=(32,100,224,224))
        op = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1), dilation=(1,1))
        y = op(x)
        self.assertEqual(y.shape, tensorshape(op, x.shape))

    def test_multiop(self):
        x = torch.rand(size=(32,100,224,224))

        ops = nn.Sequential(
            nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(3,5), stride=(2,1), padding=(5,1), dilation=(3,2), groups=4),
            nn.BatchNorm2d(num_features=200),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1), dilation=(1,1)),
            nn.Conv2d(in_channels=200, out_channels=400, kernel_size=(3,3), stride=(1,1), padding=(0,1), dilation=(1,2), groups=1),
            nn.ReLU()
        ) 

        y = ops(x)
        self.assertEqual(y.shape, tensorshape(ops, x.shape))

if __name__ == '__main__':
    unittest.main()
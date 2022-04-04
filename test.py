import unittest

import torch 
import torch.nn as nn

#from src.torchshape.shape_calculator import tensorshape

import src.torchshape as ts

class Test(unittest.TestCase):
    def test_conv2d(self):
        x = torch.rand(size=(32,100,224,224))
        op = nn.Conv2d(in_channels=100, out_channels=200, kernel_size=(3,5), stride=(2,1), padding=(5,1), dilation=(3,2), groups=4)
        y = op(x)
        self.assertEqual(y.shape, ts.tensorshape(op, x.shape))

    def test_maxpool2d(self):
        x = torch.rand(size=(32,100,224,224))
        op = nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=(1,1), dilation=(1,1))
        y = op(x)
        self.assertEqual(y.shape, ts.tensorshape(op, x.shape))

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
        self.assertEqual(y.shape, ts.tensorshape(ops, x.shape))

if __name__ == '__main__':
    unittest.main()
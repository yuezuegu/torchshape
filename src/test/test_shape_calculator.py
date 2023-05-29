import pytest
import torch

from torchshape import tensorshape


@pytest.mark.parametrize("kernel_size", [1, 2])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_tensorshape_convtranspose2d(kernel_size, stride, groups):
    op = torch.nn.ConvTranspose2d(
        in_channels=4,
        out_channels=8,
        kernel_size=kernel_size,
        stride=stride,
        groups=groups,
    )
    inshape = (1, 4, 64, 64)
    outshape = tensorshape(op, inshape)

    expected_in = torch.empty(inshape)
    expected_out = op(expected_in)
    assert expected_out.shape == outshape

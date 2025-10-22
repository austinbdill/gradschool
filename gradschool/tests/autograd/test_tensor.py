import pytest

from gradschool.autograd.tensor import Tensor


def test_shape_initialization():
    """Test valid and invalid shape initialization."""
    data = [1.0, 0.0, 2.0, 1.2]

    valid_shapes = [4, (4,), (4, 1), (4, 1, 1), (2, 2), (2, 1, 2)]
    for valid_shape in valid_shapes:
        Tensor(data, valid_shape)

    invalid_shapes = [3, (2, 3), (1, 1, 1), (-1,)]
    for invalid_shape in invalid_shapes:
        with pytest.raises(ValueError):
            Tensor(data, invalid_shape)

def test_adding_scalars():
    """Test that you can add two scalar 'tensors'."""
    a = Tensor(3.0, 1)
    b = Tensor(1.2, 1)
    assert a + b == Tensor(4.2, 1)

def test_adding_1d_tensors():
    """Test that you can add two 1D tensors."""
    a = Tensor([-1.0, 2.0, -3.0], 3)
    b = Tensor([3.0, 2.0, 1.0], 3)
    assert a + b == Tensor([2.0, 4.0, -2.0], 3)

def test_adding_multi_dimensional_tensors():
    """Test that you can add two 2D+ tensors."""
    a = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
    b = Tensor([3.0, 1.0, 2.0, 4.0], (2, 2))
    assert a + b == Tensor([4.0, 3.0, 5.0, 8.0], (2, 2))

def test_adding_different_shaped_tensors_fails():
    """Test that you can't add two tensors with different shapes."""
    # Can't add scalar and tensor (yet).
    a = Tensor(3.0, 1)
    b = Tensor([-1.0, 2.0, -3.0], 3)
    with pytest.raises(ValueError):
        a + b
    
    # Same number of elements but different shapes. 
    c = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
    d = Tensor([3.0, 1.0, 2.0, 4.0], (4,))
    with pytest.raises(ValueError):
            c + d

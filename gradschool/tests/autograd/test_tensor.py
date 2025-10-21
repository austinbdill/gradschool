
import pytest 

from gradschool.autograd.tensor import Tensor

def test_sanity():
    tensor = Tensor()
    assert True
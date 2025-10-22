from typing import List, Union
from functools import reduce
import operator


class Tensor:
    """Initial tensor implementation"""

    def __init__(
        self, data: Union[float, List[float]], shape: Union[int, Tuple[int, ...]]
    ) -> None:
        """
        :param data:
        :param shape:
        """
        num_expected_elements = (
            shape if isinstance(shape, int) else reduce(operator.mul, shape)
        )
        num_elements = 1 if isinstance(data, float) else len(data)
        if num_expected_elements != num_elements:
            raise ValueError(f"Invalid shape {shape} for data with length {len(data)}.")

        self._data = data
        self._shape = shape
        self._length = num_elements
        self._ndim = 1 if isinstance(shape, int) else len(shape)

    def __add__(self, other: Tensor) -> Tensor:
        """
        Sum data from self and other. 
        :param other: 
        :returns: 
        """
        # Need to have a more general solution for broadcasting. 
        if self._shape != other._shape:
            raise ValueError(f"Can't sum tensor with shape {self._shape} and tensor with shape {other._shape}")
        summed_data = self._data + other._data if self._length == 1 else [self._data[i] + other._data[i] for i in range(self._length)]
        return Tensor(summed_data, self._shape)

    # TODO: Needs to be tested. 
    def __eq__(self, other: Tensor) -> bool:
        """
        Check equality of two tensors. 
        :param other:
        :returns:
        """ 
        return self._shape == other._shape and self._data == other._data
from typing import List


class Tensor:
    def __init__(self, data: List[float]) -> None:
        self._data = data

    def __add__(self, other: Tensor) -> Tensor:
        pass

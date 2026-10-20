from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

@dataclass(slots=True, frozen=True)
class Transition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    done: bool

class ReplayBuffer(Sequence[Transition]):
    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("Capacity must be positive")

        self.capacity = capacity
        self._data: list[Transition] = []
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index):
        size = len(self._data)

        if isinstance(index, slice):
            return [
                self[i]
                for i in range(*index.indices(size))
            ]

        if index < 0:
            index += size
        if not 0 <= index < size:
            raise IndexError(index)
        if size < self.capacity:
            return self._data[index]

        physical_index = (self._next_idx + index) % self.capacity
        return self._data[physical_index]

    def append(self, transition: Transition) -> None:
        if len(self._data) < self.capacity:
            self._data.append(transition)
        else:
            self._data[self._next_idx] = transition

            self._next_idx = (self._next_idx + 1) % self.capacity

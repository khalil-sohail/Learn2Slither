from dataclasses import dataclass
import numpy as np

@dataclass(slots=True, frozen=True)
class Transition:
    state: np.ndarray
    action: int
    next_state: np.ndarray
    reward: float
    done: bool


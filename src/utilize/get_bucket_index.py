import random
import numpy as np

from app_config.config import GLOBAL_CONFIG

BOARD_BUCKETS = [
    # [5, 10, 15, 20], [25, 30, 35, 40], [45, 50, 55, 60], [65, 70, 75, 80], [85, 90, 95, 100]
    [5, 8, 10, 13], [15, 18, 20, 23], [25, 28, 30, 33], [35, 38, 40, 43], [50, 60, 70, 80]
]

PHASE_WEIGHTS = [
    [1.00, 0.00, 0.00, 0.00, 0.00],
    [0.30, 0.70, 0.00, 0.00, 0.00],
    [0.15, 0.15, 0.70, 0.00, 0.00],
    [0.10, 0.10, 0.10, 0.70, 0.00],
    [0.07, 0.07, 0.07, 0.07, 0.72]
]

def sample_board_size(bucket_steps: list[int], balancing_steps: int) -> tuple[int, int, bool] | None:
    """
    Samples a board size based on the current bucket steps and balancing steps.
    
    equations:
    balance_interval = minimum // 4
    balance_until = (i - 1) * balance_interval + (minimum * 2 if i == count - 1 else balance_interval)
    balancing phase weights = 1.0 / (j + 1) ** alpha if j <= i else 0.0 for j in range(count)
                    where alpha is a hyperparameter that controls the steepness of the decay.
    """
    minimum = GLOBAL_CONFIG.get("end_total_steps")
    balance_interval = max(1, minimum // 4)
    count = len(BOARD_BUCKETS)

    assert len(bucket_steps) == count
    assert len(PHASE_WEIGHTS) == count

    for i in range(count):
        if bucket_steps[i] < minimum:
            weights = PHASE_WEIGHTS[i]
            balance_until = None
            break

        if i == 0:
            continue

        balance_until = (
            (i - 1) * balance_interval
            + (minimum * 1.5 if i == count - 1 else balance_interval)
        )

        if balancing_steps < balance_until:
            alpha = 0.5
            weights = [
                1.0 / (j + 1) ** alpha if j <= i else 0.0
                for j in range(count)
            ]
            break

    else:
        return None

    bucket = random.choices(
        range(count),
        weights=weights,
        k=1,
    )[0]

    board = random.choice(BOARD_BUCKETS[bucket])
    return bucket, board, (balance_until is not None)


import random
import numpy as np

from app_config.config import GLOBAL_CONFIG

BOARD_BUCKETS = [
    # [5, 10, 15, 20], [25, 30, 35, 40], [45, 50, 55, 60], [65, 70, 75, 80], [85, 90, 95, 100]
    [5, 10], [15, 20], [25, 30], [35, 40], [45, 50]
]

PHASE_WEIGHTS = [
    [1.00, 0.00, 0.00, 0.00, 0.00],
    [0.30, 0.70, 0.00, 0.00, 0.00],
    [0.15, 0.15, 0.70, 0.00, 0.00],
    [0.10, 0.10, 0.10, 0.70, 0.00],
    [0.07, 0.07, 0.07, 0.07, 0.72]
]

def sample_board_size(bucket_steps: list[int], balancing_steps: int) -> tuple[int, int, bool] | None:
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
            + (minimum if i == count - 1 else balance_interval)
        )

        if balancing_steps < balance_until:
            weights = [
                1.0 if j <= i else 0.0
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


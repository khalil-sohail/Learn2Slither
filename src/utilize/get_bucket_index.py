import random

from app_config.config import GLOBAL_CONFIG

BOARD_BUCKETS = [
    [5, 10],
    [15, 20],
    [25, 30],
    [35, 40],
    [45, 50],
    [55, 60],
    [65, 70],
    [75, 80],
    [85, 90],
    [95, 100],
]

PHASE_WEIGHTS = [
    [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.35, 0.65, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.30, 0.10, 0.60, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.25, 0.12, 0.08, 0.55, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.22, 0.10, 0.08, 0.05, 0.55, 0.00, 0.00, 0.00, 0.00, 0.00],
    [0.16, 0.10, 0.08, 0.06, 0.05, 0.55, 0.00, 0.00, 0.00, 0.00],
    [0.15, 0.09, 0.07, 0.05, 0.05, 0.04, 0.55, 0.00, 0.00, 0.00],
    [0.14, 0.08, 0.06, 0.05, 0.04, 0.04, 0.04, 0.55, 0.00, 0.00],
    [0.13, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03, 0.55, 0.00],
    [0.12, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.03, 0.02, 0.55],
]


def sample_board_size(
    bucket_steps: list[int],
    balancing_steps: int
) -> tuple[int, int, bool] | None:
    """
    Samples a board size based on the current bucket steps and balancing steps.

    Args:
        bucket_steps:
            The number of steps taken in each bucket.
        balancing_steps:
            The number of steps taken in the current balancing phase.

    Returns:
        A tuple containing the selected bucket index, the selected board size,
        and a boolean indicating whether the balancing phase is active.
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
            + ((minimum // 2) if i == count - 1 else balance_interval)
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

import random
import numpy as np

from app_config.config import GLOBAL_CONFIG

BOARD_BUCKETS = [
    [5, 6, 7, 8, 9, 10, 12, 14, 15],
    [16, 18, 20, 23, 25, 28, 30],
    [32, 35, 40, 45, 50],
    [55, 60, 65, 70, 75],
    [80, 85, 90, 95, 100],
]

def sample_board_size(progress: float, bucket_steps: list[int]) -> tuple[int, int]:

    if progress < 0.20:
        unlocked = [0]
        target = np.array([1.0])
    elif progress < 0.40:
        unlocked = [0, 1]
        target = np.array([0.55, 0.45])
    elif progress < 0.60:
        unlocked = [0, 1, 2]
        target = np.array([0.40, 0.35, 0.25])
    elif progress < 0.80:
        unlocked = [0, 1, 2, 3]
        target = np.array([0.30, 0.27, 0.23, 0.20])
    else:
        unlocked = [0, 1, 2, 3, 4]
        target = np.array([0.20] * 5)

    unlocked_steps = np.array([bucket_steps[i] for i in unlocked])

    desired_steps = (
        GLOBAL_CONFIG.get("end_total_steps")
        * progress
        * target
    )

    deficit = np.maximum(desired_steps - unlocked_steps, 0)

    if deficit.sum() == 0:
        deficit_weights = np.ones(len(unlocked)) / len(unlocked)
    else:
        deficit_weights = deficit / deficit.sum()

    uniform_weights = np.ones(len(unlocked)) / len(unlocked)
    alpha = 0.80
    weights = alpha * deficit_weights + (1 - alpha) * uniform_weights

    bucket = random.choices(unlocked, weights=weights.tolist(), k=1)[0]
    board = random.choice(BOARD_BUCKETS[bucket])

    return bucket, board

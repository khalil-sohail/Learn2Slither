
from app_config import GLOBAL_CONFIG
from moves_engine import MovesEngine
from board_state import BoardState
from constants import Action, Entity
from gymnasium import spaces

import numpy as np


class GymnasiumWrapper:
    """Snake environment with training-only stagnation detection."""

    def __init__(self):
        self.action_space = spaces.Discrete(4)
        self.board_size = GLOBAL_CONFIG.get("board_size")

        self.green_apple_reward = GLOBAL_CONFIG.get("reward_green_apple")
        self.red_apple_reward = GLOBAL_CONFIG.get("reward_red_apple")
        self.death_reward = GLOBAL_CONFIG.get("reward_death")
        self.step_reward = GLOBAL_CONFIG.get("reward_step")

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(33,),
            dtype=np.float32,
        )

        self.board: BoardState = BoardState()
        self.physics: MovesEngine = MovesEngine()
        self.reset_training_progress()

    def reset(self, board_size: int = -1):
        """Start a new Snake episode."""
        if board_size != -1:
            self.board_size = board_size

        if GLOBAL_CONFIG.get("board_size") != self.board_size:
            GLOBAL_CONFIG.update("board_size", self.board_size)

        self.board.reset_game()
        self.physics.reset_board_size()
        self.reset_training_progress()

        self.episode_steps = 0
        self.max_episode_steps = max(1_000, self.board_size * self.board_size)

    def reset_training_progress(self):
        """
        Reset exploration tracking for a new episode
        or after successfully collecting a green apple.
        """
        self.visited_positions = {self.board.head_index}
        self.steps_without_progress = 0

        self.seen_green_since_last_apple = False
        self.best_clear_green_proximity = 0.0

    def update_training_progress(
        self,
        vision_vector: np.ndarray,
        ate_green_apple: bool
    ) -> tuple[bool, bool]:
        """
        Track genuine exploration and visible-apple progress.

        Returns:
            is_new_position
            can_see_green_apple
        """
        if ate_green_apple:
            self.reset_training_progress()
            return False, False

        position = self.board.head_index

        is_new_position = position not in self.visited_positions
        self.visited_positions.add(position)

        can_see_green_apple = any(
            vision_vector[3 + i * 8] > 0
            for i in range(4)
        )
        clear_green_proximity = max(vision_vector[6 + i * 8] for i in range(4))
        discovered_green = (
            can_see_green_apple
            and not self.seen_green_since_last_apple
        )

        if can_see_green_apple:
            self.seen_green_since_last_apple = True

        approaching_green = (
            clear_green_proximity > self.best_clear_green_proximity + 1e-6
        )
        self.best_clear_green_proximity = max(
            self.best_clear_green_proximity,
            clear_green_proximity
        )

        if (is_new_position or discovered_green or approaching_green):
            self.steps_without_progress = 0
        else:
            self.steps_without_progress += 1

        return is_new_position, can_see_green_apple

    def get_exploration_reward(
        self,
        is_new_position: bool,
        can_see_green_apple: bool
    ) -> float:
        """
        Encourage visiting new cells when no green apple
        is visible.

        The bonus remains smaller than the step penalty.
        """
        if is_new_position and not can_see_green_apple:
            return 0.005

        return 0.0

    def should_stop_training(self) -> bool:
        """
        Truncate training after prolonged stagnation.

        This is not a Snake death condition.
        """
        timeout_limit = max(100, int(self.board_size * 3))
        return (
            self.steps_without_progress >= timeout_limit
            or self.episode_steps >= self.max_episode_steps
        )

    def calculate_reward(
        self,
        is_dead: bool,
        ate_red_apple: bool,
        ate_green_apple: bool
    ) -> float:
        """Calculate normal game rewards without starvation."""
        reward = self.step_reward

        if ate_green_apple:
            reward += self.green_apple_reward

        if ate_red_apple:
            reward += self.red_apple_reward

        if is_dead:
            reward += self.death_reward

        return reward

    def step(
        self,
        action: Action,
        training: bool = False
    ) -> tuple[np.ndarray, float, bool, bool]:
        """
        Execute a Snake movement.

        Returns:
            observation, reward, is_dead, stop_training

        is_dead:
            Genuine game termination.

        stop_training:
            Training-only truncation; not treated as death.
        """
        self.episode_steps += 1
        next_index = self.physics.get_next_position(
            self.board.head_index,
            action
        )
        is_dead = self.physics.check_wall_body_collision(
            next_index,
            self.board.get_board_array()
        )

        ate_red_apple = False
        ate_green_apple = False
        if not is_dead:
            ate_apple = self.board.move_snake(next_index)
            ate_red_apple = ate_apple == Entity.RED_APPLE
            ate_green_apple = ate_apple == Entity.GREEN_APPLE

            if not self.board.body:
                is_dead = True

        if is_dead:
            vision_vector = np.array([0.0] * 33, dtype=np.float32)
        else:
            vision_vector = self.board.get_snake_vision_vector(action)

        reward = self.calculate_reward(is_dead, ate_red_apple, ate_green_apple)
        stop_training = False
        if training and not is_dead:
            is_new_position, can_see_green_apple = (
                self.update_training_progress(
                    vision_vector,
                    ate_green_apple,
                )
            )

            if not ate_green_apple:
                reward += self.get_exploration_reward(
                    is_new_position,
                    can_see_green_apple,
                )

            stop_training = self.should_stop_training()

        return vision_vector, reward, is_dead, stop_training

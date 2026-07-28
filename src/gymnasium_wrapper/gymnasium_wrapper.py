# this is a wrapper for gymnasium.
# the only purpose of this wrapper is to create a gymnasium environment for the snake game,
#     allowing the agent/user to interact with the game in a standardized way.

# the only allowed actions are:
# 0 - move up
# 1 - move down
# 2 - move left
# 3 - move right


import math

from app_config.config import GLOBAL_CONFIG
from moves_engine import MovesEngine
from board_state import BoardState
from constants import Action, Entity
from gymnasium import spaces
import gymnasium as gym
import numpy as np

class GymnasiumWrapper(gym.Env):
    """A Gymnasium wrapper for the Snake game."""

    def __init__(self):
        """
        Initializes the Gymnasium environment for the Snake game.
        :param board_size: The size of the board (number of cells in one dimension).
        """
        self.action_space = spaces.Discrete(4)
        self.board_size = GLOBAL_CONFIG.get('board_size')
        self.observation_space = spaces.Box(low=0, high=4, shape=(self.board_size * self.board_size,), dtype=np.int32)
        self.board = BoardState()
        self.physics = MovesEngine()
        self.steps_since_last_green_apple = 0

    def reset(self):
        """
        Resets the environment to an initial state.
        :return: The initial observation.
        """
        self.observation_space = spaces.Box(low=0, high=4, shape=(self.board_size * self.board_size,), dtype=np.int32)
        self.steps_since_last_green_apple = 0
        self.board_size = GLOBAL_CONFIG.get('board_size')
        self.board.reset_game()
        self.physics.reset_board_size()
        return self.board.get_snake_vision_vector()

    def step(self, action: Action):
        """
        Executes one time step within the environment based on the given action.
        :param action: The action to be taken (Action.UP: Up, Action.DOWN: Down, Action.LEFT: Left, Action.RIGHT: Right).
        :return: A tuple containing the next observation, reward, terminated flag.
        """
        next_index = self.physics.get_next_position(self.board.head_index, action)
        is_dead = self.physics.check_wall_body_collision(next_index, self.board.get_board_array())
        
        if not is_dead:
            ate_apple = self.board.move_snake(next_index)
            ate_red_apple = ate_apple == Entity.RED_APPLE
            ate_green_apple = ate_apple == Entity.GREEN_APPLE
            if ate_red_apple and len(self.board.body) <= 0:
                is_dead = True
        else:
            ate_red_apple = ate_green_apple = False

        reward, done = self._calculate_reward(is_dead, ate_red_apple, ate_green_apple)

        return self.board.get_snake_vision_vector(), reward, done or is_dead

    def _calculate_reward(self, is_dead: bool, ate_red_apple: bool, ate_green_apple: bool):
        """Calculates the reward for the current step."""
        done = False
        reward = 0
        
        if ate_red_apple:
            reward += GLOBAL_CONFIG.get("reward_red_apple")

        if ate_green_apple:
            self.steps_since_last_green_apple = 0
            reward += GLOBAL_CONFIG.get("reward_green_apple")
        else:
            self.steps_since_last_green_apple += 1

        if is_dead:
            reward += GLOBAL_CONFIG.get("reward_death")

        board_size = self.board_size
        head = self.board.head_index

        def manhattan(index1, index2):
            x1, y1 = index1 % board_size, index1 // board_size
            x2, y2 = index2 % board_size, index2 // board_size
            return abs(x1 - x2) + abs(y1 - y2)

        distance = min(
            manhattan(head, self.board.green_apple[0]),
            manhattan(head, self.board.green_apple[1]),
        )
        ratio = self.steps_since_last_green_apple / max(distance, 1)
        reward_step = -0.01 - min((ratio / 10.0) ** 2, 1.0)
            
        s = self.steps_since_last_green_apple
        reward_step = -0.01 - min(
            (s / 100.0) ** 2 * 0.5,
            1.0
        )

        return reward + reward_step, done


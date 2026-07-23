# this is a wrapper for gymnasium.
# the only purpose of this wrapper is to create a gymnasium environment for the snake game,
#     allowing the agent/user to interact with the game in a standardized way.

# the only allowed actions are:
# 0 - move up
# 1 - move down
# 2 - move left
# 3 - move right


from app_config.config import GLOBAL_CONFIG
from moves_engine import MovesEngine
from board_state import BoardState
from constants import Entity
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
        self.observation_space = spaces.Box(low=0, high=4, shape=(GLOBAL_CONFIG.get('board_size') * GLOBAL_CONFIG.get('board_size'),), dtype=np.int32)
        self.board = BoardState()
        self.physics = MovesEngine()

    def reset(self):
        """
        Resets the environment to an initial state.
        :return: A tuple containing the initial observation and an empty info dictionary.
        """
        self.observation_space = spaces.Box(low=0, high=4, shape=(GLOBAL_CONFIG.get('board_size') * GLOBAL_CONFIG.get('board_size'),), dtype=np.int32)
        # physics = MovesEngine() # no need for now
        self.board.reset_game()
        return self.board.get_board_array(), {}

    def step(self, action: int):
        """
        Executes one time step within the environment based on the given action.
        :param action: The action to be taken (Action.UP: Up, Action.DOWN: Down, Action.LEFT: Left, Action.RIGHT: Right).
        :return: A tuple containing the next observation, reward, terminated flag.
        """
        next_index = self.physics.get_next_position(self.board.head_index, action)
        is_dead = self.physics.check_wall_body_collision(next_index, self.board.get_board_array())
        ate_red_apple = self.physics.check_apple_collision(self.board.red_apple, next_index)
        ate_green_apple = self.physics.check_apple_collision(self.board.green_apple, next_index)

        if not is_dead:
            self.board.move_snake(next_index)
            if ate_red_apple and len(self.board.body) <= 0:
                is_dead = True

        reward = self._calculate_reward(is_dead, ate_red_apple, ate_green_apple)
        # truncated = False
        # info = {}
        tail_direction = self.board.get_tail_direction(self.board.body[0], self.board.body[1]) if len(self.board.body) > 1 else None
        
        return self.board.get_board_array(), reward, is_dead, tail_direction

    def _calculate_reward(self, is_dead: bool, ate_red_apple: bool, ate_green_apple: bool):
        """Calculates the reward for the current step."""
        reward = 0.0

        if is_dead:
            reward += GLOBAL_CONFIG.get('reward_death')
        if ate_red_apple:
            reward += GLOBAL_CONFIG.get('reward_red_apple')
        if ate_green_apple:
            reward += GLOBAL_CONFIG.get('reward_green_apple')

        return reward

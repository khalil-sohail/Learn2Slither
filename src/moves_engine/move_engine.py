from app_config import GLOBAL_CONFIG
from constants import Action,  Entity

import numpy as np


class MovesEngine:
    """
    The MovesEngine class is responsible for handling the movement
        logic of the snake in the game.
    It calculates the next position of the snake's head based on the current
        action and checks for collisions with walls or the snake's own body.
    """
    def __init__(self):
        """
        Initializes the MovesEngine with the specified board size.
        :param size: The size of the board (number of cells in one dimension).
        """
        self.board_size = GLOBAL_CONFIG.get("board_size")

    def reset_board_size(self):
        """
        Resets the board size from the GLOBAL_CONFIG.
        This is useful if the board size is changed during runtime.
        """
        self.board_size = GLOBAL_CONFIG.get("board_size")

    def get_next_position(self, head_index, action):
        """
        Returns the hypothetical next integer index based on:
            Up - Down - Left - Right actions.
        :param head_index: The current index of the snake's head.
        :param action: The action to be taken (
            Action.UP: Up,
            Action.DOWN: Down,
            Action.LEFT: Left,
            Action.RIGHT: Right
        ).
        :return: The next index after applying the action.
        """
        row = head_index // self.board_size
        col = head_index % self.board_size

        if action == Action.UP:
            row = (row - 1) % self.board_size
        elif action == Action.DOWN:
            row = (row + 1) % self.board_size
        elif action == Action.LEFT:
            col = (col - 1) % self.board_size
        elif action == Action.RIGHT:
            col = (col + 1) % self.board_size

        return row * self.board_size + col

    def check_wall_body_collision(
        self,
        next_index: int,
        board_array: np.ndarray
    ) -> bool:
        """
        Checks if the next_index collides with the wall or the snake's body.
        :param next_index: The index to check for collision.
        :param board_array: The current board state as a 1D array.
        :return: True if there is a collision, False otherwise.
        """
        if next_index < 0 or next_index >= len(board_array):
            return True

        destination_value = board_array[next_index]
        if destination_value in [Entity.WALL, Entity.SNAKE_BODY]:
            return True
        return False

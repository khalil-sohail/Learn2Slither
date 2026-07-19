# this is the main file for the board state.
#      it will be used to store the game state/board in both training and playing scenarios.
#      it will also be used to watch the training process.

# the board will be represented as a 1D array of size board^2, where each cell can have the following values:
# 0 - empty cell
# 1 - snake body
# 2 - snake head
# 3 - red apple
# 4 - green apple

# the Board is also capable of returning what the snake sees in its current state.
#     which will be used as input for the neural network.
#     the snake sees in 4 directions (N, E, S, W) a plus (+) like shape unlise blocked by a wall or the snake's own body.

# the snake vision, what the agent sees
#     will construct as a strictly fixed array of 16 numbers
#     for each of the 4 directions (North, South, East, West)
#     it will return the following information:
#         - Distance to Wall (Inverted to v=1/d for normalization)
#         - Distance to Nearest Body Segment (v=1/d, or 0 if none)
#         - Distance to Nearest Green Apple (v=1/d, or 0 if none)
#         - Distance to Nearest Red Apple (v=1/d, or 0 if none)
# so we have 4 cardinal directions (North, South, East, West), and to calculate 4 specific distance features.
#    we will have a total of 16 features for the snake vision.

from collections import deque
from constants import Entity

import numpy as np

class BoardState:
    """
    Represents the state of the game board.
    The board is represented as a 1D array of size board_size^2, where each cell can have the following values:
        0 - empty cell
        1 - snake body
        2 - snake head
        3 - red apple
        4 - green apple
    
    args:
        size: The size of the board (number of cells in one dimension).
    
    functions:
        reset_game(): Resets the game state, clearing the board and placing the initial snake head and apples.
        move_snake(next_index): Moves the snake to the specified index, updating the board state accordingly.
        spawn_apple(color): Spawns an apple of the specified color (3 for red, 4 for green) at a random empty cell.
        get_vision_vector(): Returns a 16-value float32 NumPy array representing what the snake sees in its current state.
        get_terminal_vision_string(): Returns a strictly formatted plus-shape string representation of the snake's vision for terminal display.
    """
    def __init__(self, size):
        """
        Initializes the BoardState with the specified size.
        :param size: The size of the board (number of cells in one dimension).
        """
        self.size = size
        self.board = np.zeros((size*size), dtype=int)
        self.body = deque()
        self.red_apple = None
        self.green_apple = None
        self.head_index = None

    def reset_game(self):
        """
        Resets the game state, clearing the board and placing the initial snake head and apples.
        """
        self.board.fill(Entity.EMPTY)
        self.board[:self.size] = Entity.WALL
        self.board[-self.size:] = Entity.WALL
        self.board[::self.size] = Entity.WALL
        self.board[self.size - 1::self.size] = Entity.WALL

        self.body.clear()
        self.body.append((self.size*self.size // 2) - self.size // 2)
        self.head_index = (self.size*self.size // 2) - self.size // 2

        self.spawn_apple(3)
        self.spawn_apple(4)
        self.spawn_apple(4)

    def move_snake(self, next_index):
        """
        Moves the snake to the specified index, updating the board state accordingly.
        :param next_index: The index to which the snake's head will move.
        """
        ate_apple = self.board[next_index]

        if self.body:
            current_head_index = self.body[-1]
            self.board[current_head_index] = Entity.SNAKE_BODY
        self.body.append(next_index)
        self.board[next_index] = Entity.SNAKE_HEAD

        if ate_apple == Entity.RED_APPLE:
            self.red_apple = None
            self.spawn_apple(Entity.RED_APPLE)
            tail_index = self.body.popleft() ### need a better way for that
            self.board[tail_index] = Entity.EMPTY
            tail_index = self.body.popleft()
            self.board[tail_index] = Entity.EMPTY
        elif ate_apple == Entity.GREEN_APPLE:
            self.green_apple = None
            self.spawn_apple(Entity.GREEN_APPLE)
        else:
            tail_index = self.body.popleft()
            self.board[tail_index] = Entity.EMPTY

        self.head_index = next_index

    def spawn_apple(self, color):
        """
        Spawns an apple of the specified color (3 for red, 4 for green) at a random empty cell.
        """
        random_empty_indices = np.where(self.board == 0)[0]
        if random_empty_indices.size > 0:
            random_index = np.random.choice(random_empty_indices)
            self.board[random_index] = color
            if color == 3:
                self.red_apple = random_index
            elif color == 4:
                self.green_apple = random_index

    def get_board_array(self):
        """
        Returns a copy of the current board state as a 1D NumPy array."""
        return self.board.copy()

    def get_vision_vector(self):
        """
        Calculates and returns the 16-value float32 NumPy array representing what the snake sees in its current state.
        The vision vector consists of 4 features for each of the 4 cardinal directions (North, East, South, West):
            - Distance to Wall (Inverted to v=1/d for normalization)
            - Distance to Nearest Body Segment (v=1/d, or 0 if none)
            - Distance to Nearest Green Apple (v=1/d, or 0 if none)
            - Distance to Nearest Red Apple (v=1/d, or 0 if none)
        :return: A 16-value float32 NumPy array representing the snake's vision.
        """
        pass

    def get_terminal_vision_string(self):
        """
        Calculates and returns a strictly formatted plus-shape string representation of the snake's vision for terminal display.
        The vision string will visually represent the snake's vision in a plus-shape format, with the snake's head at the center and the distances to walls, body segments, and apples in the four cardinal directions.
        :return: A string representing the snake's vision in a plus-shape format for terminal display.
        """
        pass

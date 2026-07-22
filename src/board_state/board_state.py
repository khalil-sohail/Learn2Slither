# this is the main file for the board state.
#      it will be used to store the game state/board in both training and playing scenarios.
#      it will also be used to watch the training process.

# the board will be represented as a 1D array of size board^2, where each cell can have the following values:
# 6 - empty cell
# 3 - snake head
# 4 - snake body
# 5 - snake tail
# 7 - red apple
# 8 - green apple
# 9 - wall

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
from app_config.config import GLOBAL_CONFIG
from constants import Action, Entity, Textures

import numpy as np

class BoardState:
    """
    Represents the state of the game board.
    The board is represented as a 1D array of size board_size^2, where each cell can have the following values:
        6 - empty cell
        3 - snake head
        4 - snake body
        5 - snake tail
        7 - red apple
        8 - green apple
        9 - wall
    
    args:
        size: The size of the board (number of cells in one dimension).
    
    functions:
        reset_game(new_size: int): Resets the game state, clearing the board and placing the initial snake head and apples.
        move_snake(next_index: int): Moves the snake to the specified index, updating the board state accordingly.
        spawn_apple(color: Entity): Spawns an apple of the specified color (Entity.RED_APPLE for red, Entity.GREEN_APPLE for green) at a random empty cell.
        get_vision_vector(): Returns a 16-value float32 NumPy array representing what the snake sees in its current state.
        get_terminal_vision_string(): Returns a strictly formatted plus-shape string representation of the snake's vision for terminal display.
    """
    def __init__(self):
        """
        Initializes the BoardState with the specified size.
        :param size: The size of the board (number of cells in one dimension).
        """
        self.size = GLOBAL_CONFIG.get('board_size')
        self.board = np.zeros((self.size*self.size), dtype=int)
        self.body = deque()
        self.red_apple = deque()
        self.green_apple = deque()
        self.red_apple_eaten = 0
        self.green_apple_eaten = 0
        self.board_score = 0
        self.head_index = None
        self.tail_direction = Textures.SNAKE_TAIL_LEFT

    def reset_game(self):
        """
        Resets the game state, clearing the board and placing the initial snake head and apples.
        """
        self.size = GLOBAL_CONFIG.get('board_size')
        self.board = np.zeros((self.size*self.size), dtype=int)
        self.body.clear()
        self.red_apple.clear()
        self.green_apple.clear()
        self.red_apple_eaten = 0
        self.green_apple_eaten = 0
        self.board_score = 0
        self.head_index = None
        self.tail_direction = Textures.SNAKE_TAIL_LEFT

        self.board.fill(Entity.EMPTY)
        self.board[:self.size] = Entity.WALL
        self.board[-self.size:] = Entity.WALL
        self.board[::self.size] = Entity.WALL
        self.board[self.size - 1::self.size] = Entity.WALL

        self.body.clear()
        self.body.append((self.size*self.size // 2) - self.size // 2)
        self.head_index = (self.size*self.size // 2) - self.size // 2

        self.spawn_apple(Entity.RED_APPLE)
        self.spawn_apple(Entity.RED_APPLE)
        self.spawn_apple(Entity.RED_APPLE)
        self.spawn_apple(Entity.RED_APPLE)
        self.spawn_apple(Entity.GREEN_APPLE)
        self.spawn_apple(Entity.GREEN_APPLE)
        self.spawn_apple(Entity.GREEN_APPLE)
        self.spawn_apple(Entity.GREEN_APPLE)

    def move_snake(self, next_index: int):
        """
        Moves the snake to the specified index, updating the board state accordingly.
        :param next_index: The index to which the snake's head will move.
        """
        ate_apple = self.board[next_index]

        if self.body:
            self.board[self.body[-1]] = Entity.SNAKE_BODY

        self.body.append(next_index)
        self.board[next_index] = Entity.SNAKE_HEAD
        tail_moves = 1

        if ate_apple == Entity.RED_APPLE:
            self.red_apple.popleft()
            self.spawn_apple(Entity.RED_APPLE)
            self.red_apple_eaten += 1
            self.board_score -= 1 if self.board_score > 0 else 0
            tail_moves += 1
        elif ate_apple == Entity.GREEN_APPLE:
            self.green_apple.popleft()
            self.spawn_apple(Entity.GREEN_APPLE)
            self.green_apple_eaten += 1
            self.board_score += 1
            tail_moves -= 1

        for _ in range(tail_moves):
            self.board[self.body.popleft()] = Entity.EMPTY

        self.head_index = next_index
        if len(self.body) >= 2:
            self.board[self.body[0]] = Entity.SNAKE_TAIL

    def spawn_apple(self, color: Entity):
        """
        Spawns an apple of the specified color (Entity.RED_APPLE for red, Entity.GREEN_APPLE for green) at a random empty cell.
        """
        random_empty_indices = np.where(self.board == Entity.EMPTY)[0]
        if random_empty_indices.size > 0:
            random_index = np.random.choice(random_empty_indices)
            self.board[random_index] = color
            if color == Entity.RED_APPLE:
                self.red_apple.append(random_index)
            elif color == Entity.GREEN_APPLE:
                self.green_apple.append(random_index)

    def get_board_array(self):
        """
        Returns a copy of the current board state as a 1D NumPy array."""
        return self.board.copy()

    def get_board_statistics(self):
        """
        Returns the count of red and green apples eaten.
        :return: A tuple containing the count of red and green apples eaten.
        """
        return self.board_score, self.green_apple_eaten, self.red_apple_eaten

    def get_tail_direction(self, front_index: int, behind_index: int):
        difference = front_index - behind_index
        
        if difference == -self.size:
            return Textures.SNAKE_TAIL_UP
        elif difference == self.size:
            return Textures.SNAKE_TAIL_DOWN
        elif difference == -1:
            return Textures.SNAKE_TAIL_LEFT
        elif difference == 1:
            return Textures.SNAKE_TAIL_RIGHT
            
        return Textures.SNAKE_TAIL_LEFT

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

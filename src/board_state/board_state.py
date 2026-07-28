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
from constants import Action, Entity

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
        self.board_size = GLOBAL_CONFIG.get('board_size')
        self.board = np.zeros((self.board_size * self.board_size), dtype=int)
        self.body = deque()
        self.red_apple = deque()
        self.green_apple = deque()
        self.red_apple_eaten = 0
        self.green_apple_eaten = 0
        self.board_score = 0
        self.head_index: int = 0
        self.tail_direction = Entity.SNAKE_TAIL_LEFT

    def reset_game(self):
        """
        Resets the game state, clearing the board and placing the initial snake head and apples.
        """
        self.board_size = GLOBAL_CONFIG.get('board_size')
        self.board = np.zeros((self.board_size * self.board_size), dtype=int)
        self.body.clear()
        self.red_apple.clear()
        self.green_apple.clear()
        self.red_apple_eaten = 0
        self.green_apple_eaten = 0
        self.board_score = 0
        self.tail_direction = Entity.SNAKE_TAIL_LEFT

        self.board.fill(Entity.EMPTY)
        self.board[:self.board_size] = Entity.WALL
        self.board[-self.board_size:] = Entity.WALL
        self.board[::self.board_size] = Entity.WALL
        self.board[self.board_size - 1::self.board_size] = Entity.WALL

        interior_size = self.board_size - 2
        center_row = 1 + interior_size // 2
        center_col = 1 + interior_size // 2
        self.head_index = center_row * self.board_size + center_col
        self.body.append(self.head_index)
        self.board[self.head_index] = Entity.SNAKE_HEAD_UP
        
        self.spawn_apple(Entity.RED_APPLE)
        for i in range(2):
            self.spawn_apple(Entity.GREEN_APPLE)

    def move_snake(self, next_index: int):
        """
        Moves the snake to the specified index, updating the board state accordingly.
        :param next_index: The index to which the snake's head will move.
        :return: The Entity that was at next_index before the move (EMPTY, RED_APPLE, or GREEN_APPLE).
        """
        ate_apple = self.board[next_index]

        if self.body:
            self.board[self.body[-1]] = Entity.SNAKE_BODY

        self.body.append(next_index)
        self.board[next_index] = self.get_head_direction(self.head_index, next_index)
        tail_moves = 1

        if ate_apple == Entity.RED_APPLE:
            self.red_apple.remove(next_index)
            self.spawn_apple(Entity.RED_APPLE)
            self.red_apple_eaten += 1
            self.board_score -= 1 if self.board_score > 0 else 0
            tail_moves += 1
        elif ate_apple == Entity.GREEN_APPLE:
            self.green_apple.remove(next_index)
            self.spawn_apple(Entity.GREEN_APPLE)
            self.green_apple_eaten += 1
            self.board_score += 1
            tail_moves -= 1

        for _ in range(tail_moves):
            self.board[self.body.popleft()] = Entity.EMPTY

        self.head_index = next_index
        if len(self.body) >= 2:
            self.board[self.body[0]] = self.get_tail_direction(self.body[0], self.body[1])

        return ate_apple

    def spawn_apple(self, color: Entity):
        """
        Spawns an apple of the specified color (Entity.RED_APPLE for red, Entity.GREEN_APPLE for green) at a random empty cell.
        """
        random_empty_indices = np.where(self.board == Entity.EMPTY)[0]
        if random_empty_indices.size > 0:
            random_index = np.random.choice(random_empty_indices)
            if color == Entity.RED_APPLE:
                self.board[random_index] = Entity.RED_APPLE
                self.red_apple.append(random_index)
            elif color == Entity.GREEN_APPLE:
                self.board[random_index] = Entity.GREEN_APPLE
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
        return (
            self.board_score,
            self.green_apple_eaten,
            self.red_apple_eaten,
            self.board.copy()
        )

    def get_tail_direction(self, front_index: int, behind_index: int):
        difference = front_index - behind_index
        
        if difference == -self.board_size:
            return Entity.SNAKE_TAIL_UP
        elif difference == self.board_size:
            return Entity.SNAKE_TAIL_DOWN
        elif difference == -1:
            return Entity.SNAKE_TAIL_LEFT
        elif difference == 1:
            return Entity.SNAKE_TAIL_RIGHT
            
        return Entity.SNAKE_TAIL_LEFT

    def get_head_direction(self, front_index: int, behind_index: int):
        difference = front_index - behind_index
        
        if difference == self.board_size:
            return Entity.SNAKE_HEAD_UP
        elif difference == -self.board_size:
            return Entity.SNAKE_HEAD_DOWN
        elif difference == 1:
            return Entity.SNAKE_HEAD_LEFT
        elif difference == -1:
            return Entity.SNAKE_HEAD_RIGHT
            
        return Entity.SNAKE_HEAD_LEFT

    def get_snake_vision_vector(self):
        """
        Calculates and returns the 16-value float32 NumPy array representing what the snake sees in its current state.
        The vision vector consists of 4 features for each of the 4 cardinal directions (North, East, South, West):
            - Distance to Wall (Inverted to v=1/d for normalization)
            - Distance to Nearest Body Segment (v=1/d, or 0 if none)
            - Distance to Nearest Green Apple (v=1/d, or 0 if none)
            - Distance to Nearest Red Apple (v=1/d, or 0 if none)
        :return: A 16-value float32 NumPy array representing the snake's vision.
        """
        vision_vector = np.zeros(16, dtype=np.float32)
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # North, East, South, West
        head_row = self.head_index // self.board_size
        head_col = self.head_index % self.board_size
        
        rays_chars = {
            (-1, 0): [],
            (0, 1): [],
            (1, 0): [],
            (0, -1): []
        }

        for i, (dr, dc) in enumerate(directions):
            distance_to_wall = 0
            distance_to_body = 0
            distance_to_green_apple = 0
            distance_to_red_apple = 0

            row, col = head_row, head_col
            while True:
                row += dr
                col += dc
                distance_to_wall += 1

                if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
                    rays_chars[(dr, dc)].append("W")
                    break

                index = row * self.board_size + col
                cell_value = self.board[index]

                if cell_value in (Entity.SNAKE_BODY, Entity.SNAKE_TAIL_UP, Entity.SNAKE_TAIL_DOWN, Entity.SNAKE_TAIL_LEFT, Entity.SNAKE_TAIL_RIGHT):
                    rays_chars[(dr, dc)].append("S")
                    if distance_to_body == 0:
                        distance_to_body = distance_to_wall
                elif cell_value == Entity.GREEN_APPLE:
                    rays_chars[(dr, dc)].append("G")
                    if distance_to_green_apple == 0:
                        distance_to_green_apple = distance_to_wall
                elif cell_value == Entity.RED_APPLE:
                    rays_chars[(dr, dc)].append("R")
                    if distance_to_red_apple == 0:
                        distance_to_red_apple = distance_to_wall
                elif cell_value == Entity.WALL:
                    rays_chars[(dr, dc)].append("W")
                    break
                else:
                    rays_chars[(dr, dc)].append("0")

            vision_vector[i * 4] = 1.0 / distance_to_wall if distance_to_wall > 0 else 0.0
            vision_vector[i * 4 + 1] = 1.0 / distance_to_body if distance_to_body > 0 else 0.0
            vision_vector[i * 4 + 2] = 1.0 / distance_to_green_apple if distance_to_green_apple > 0 else 0.0
            vision_vector[i * 4 + 3] = 1.0 / distance_to_red_apple if distance_to_red_apple > 0 else 0.0

        if GLOBAL_CONFIG.get("print_vision"):
            west_str = "".join(reversed(rays_chars[(0, -1)]))
            east_str = "".join(rays_chars[(0, 1)])
            padding = " " * len(west_str)
        
            print() 
            for char in reversed(rays_chars[(-1, 0)]):
                print(padding + char)
            print(west_str + "H" + east_str)
            for char in rays_chars[(1, 0)]:
                print(padding + char)
            print() 

        return vision_vector

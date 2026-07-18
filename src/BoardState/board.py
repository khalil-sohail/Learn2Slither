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

import numpy as np

class Board:
    def __init__(self, size):
        self.size = size
        self.board = np.zeros((size*size), dtype=int)  # Initialize a 1D array of zeros with size^2 elements

    def reset(self):
        """Reset the board to its initial state."""
        self.board.fill(0)

    def updateCell(self, x, y, value):
        """Update the value of a specific cell in the board."""
        if 0 <= x < self.size and 0 <= y < self.size:
            self.board[x * self.size + y] = value
        else:
            raise IndexError("Cell coordinates are out of bounds.")

    def getCell(self, x, y):
        """Get the value of a specific cell in the board."""
        if 0 <= x < self.size and 0 <= y < self.size:
            return self.board[x * self.size + y]
        else:
            raise IndexError("Cell coordinates are out of bounds.")

    def getBoard(self):
        """Get the current state of the board."""
        return self.board
    
    def getBoard_2d(self):
        """Get the current state of the board as a 2D array."""
        return self.board.reshape((self.size, self.size))
    
    def getSnakeVision(self, snake_head, snake_body):
        """
        Get what the snake sees in its current state.
        The snake sees in 4 directions (N, E, S, W) a plus (+) like shape until blocked by a wall or the snake's own body.
        """
        pass  # Placeholder for the actual implementation of snake vision logic

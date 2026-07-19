# this is the main file for the moves engine.
# every move will be calculated and returned by this engine. that will update the board state.
# it will also be used to calculate the next state of the board after a move is made.

# the only possible moves are:
# 0 - move up
# 1 - move down
# 2 - move left
# 3 - move right

# MovesEngine engine will have two modes, the first mode is the training mode, where the moves will be calculated by the neural network.
# the second mode is the playing mode, where the moves will be calculated by the user input

# all moves will be validated before being applied to the board state, and if the move is invalid, the game will end.
# all moves will be recived from the Gymnasium environment, and the moves will be applied to the board state, and the board state will be updated accordingly.

from constants import Action

class MovesEngine:
    """
    The MovesEngine class is responsible for calculating and validating moves in the game.
    it will update the board state based on the moves made by the agent or user input.
    """
    def __init__(self, size: int):
        """
        Initializes the MovesEngine with the specified board size.
        :param size: The size of the board (number of cells in one dimension).
        """
        self.size = size

    def get_next_position(self, head_index, action):
        """
        Returns the hypothetical next integer index based on Up/Down/Left/Right action.
        :param head_index: The current index of the snake's head.
        :param action: The action to be taken (0: Up, 1: Down, 2: Left, 3: Right).
        :return: The next index after applying the action.
        """
        row = head_index // self.size
        col = head_index % self.size

        if action == Action.UP:  # Move up
            row = (row - 1) % self.size
        elif action == Action.DOWN:  # Move down
            row = (row + 1) % self.size
        elif action == Action.LEFT:  # Move left
            col = (col - 1) % self.size
        elif action == Action.RIGHT:  # Move right
            col = (col + 1) % self.size

        return row * self.size + col

    def check_wall_body_collision(self, next_index, board_array):
        """
        Checks if the next_index collides with the wall or the snake's body.
        :param next_index: The index to check for collision.
        :param board_array: The current board state as a 1D array.
        :return: True if there is a collision, False otherwise.
        """
        if next_index < 0 or next_index >= len(board_array):
            return True

        destination_value = board_array[next_index]
        
        if destination_value == 1 or destination_value == 5:
            return True
        return False

    def check_apple_collision(self, apple_index, next_index):
        """
        Checks if the next_index collides with an apple.
        :param apple_index: The index of the apple to check for collision.
        :param next_index: The index to check for collision.
        :return: True if the apple is eaten, False otherwise.
        """
        if apple_index is not None and next_index == apple_index:
            return True
        return False
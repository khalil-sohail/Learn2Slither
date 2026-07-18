# this is the main file for the moves engine.
# every move will be calculated and returned by this engine. that will update the board state.
# it will also be used to calculate the next state of the board after a move is made.

# the only possible moves are:
# 0 - move up
# 1 - move down
# 2 - move left
# 3 - move right

# MoveEngine engine will have two modes, the first mode is the training mode, where the moves will be calculated by the neural network.
# the second mode is the playing mode, where the moves will be calculated by the user input

# all moves will be validated before being applied to the board state, and if the move is invalid, the game will end.
# all moves will be recived from the Gymnasium environment, and the moves will be applied to the board state, and the board state will be updated accordingly.

class MoveEngine:
    pass
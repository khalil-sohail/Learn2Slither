# this file is responsible for rendering the game state using pygame-ce.
# It provides a GraphicsEngine class that initializes the graphics engine,
#    renders frames based on the board state,
#    and handles closing the graphics engine safely.

# the board will be represented as a 1D array of size board^2, where each cell can have the following values:
# 0 - empty cell
# 1 - snake body
# 2 - snake head
# 3 - red apple
# 4 - green apple
# 5 - wall

import pygame

class GraphicsEngine:
    def __init__(self, window_size, board_size, fps):
        """
        Initializes the graphics engine with the specified window size, board size, and frames per second (fps).
        :param window_size: The size of the window in pixels (width and height).
        :param board_size: The size of the board (number of cells in one dimension).
        :param fps: The frames per second for rendering.
        """
        pygame.init()
        self.window_size = window_size
        self.board_size = board_size
        self.fps = fps
        self.screen = pygame.display.set_mode((window_size, window_size))
        self.clock = pygame.time.Clock()

    def render_frame(self, board_array):
        """
        Renders the current board state to the display surface.
        :param board_array: A 1D NumPy array representing the current board state.
        """
        self.screen.fill((0, 0, 0))

        cell_size = self.window_size // self.board_size
        for index, value in enumerate(board_array):
            row = index // self.board_size
            col = index % self.board_size
            x = col * cell_size
            y = row * cell_size

            if value == 0:
                color = (0, 0, 0)
            elif value == 1:
                color = (255, 255, 255)
            elif value == 2:
                color = (222, 222, 222)
            elif value == 3:
                color = (255, 0, 0)
            elif value == 4:
                color = (0, 0, 255)
            elif value == 5:
                color = (100, 100, 100)
            
            if value != 0:
                pygame.draw.rect(self.screen, color, (x, y, cell_size, cell_size))

        pygame.display.flip()
        self.clock.tick(self.fps)
        
    def close(self):
        """
        Safely closes the graphics engine and quits pygame.
        """
        pygame.quit()
    
    def game_over(self):
        """
        Displays a "Game Over" message on the screen.
        """
        font = pygame.font.SysFont(None, 100)
        text = font.render('Game Over', True, (255, 255, 0))
        text_rect = text.get_rect(center=(self.window_size // 2, self.window_size // 2))
        self.screen.blit(text, text_rect)
        pygame.display.flip()
        pygame.time.wait(500)
# this file is responsible for rendering the game state using pygame-ce.
# It provides a GraphicsEngine class that initializes the graphics engine,
#    renders frames based on the board state,
#    and handles closing the graphics engine safely.

# the board will be represented as a 1D array of size board^2, where each cell can have the following values:
# 6 - empty cell
# 3 - snake head
# 4 - snake body
# 5 - snake tail
# 7 - red apple
# 8 - green apple
# 9 - wall

from constants import Entity, Textures
import pygame
import os

class GraphicsEngine:
    def __init__(self, window_size, board_size, fps):
        """
        Initializes the graphics engine with the specified window size, board size, and frames per second (fps).
        :param window_size: The size of the window in pixels (width and height).
        :param board_size: The size of the board (number of cells in one dimension).
        :param fps: The frames per second for rendering.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((window_size, window_size))
        self.clock = pygame.time.Clock()

        self.window_size = window_size
        self.board_size = board_size
        self.cell_size = window_size // board_size
        self.fps = fps
        
        self.textures = self._load_textures()

    def _load_textures(self):
        textures = {}
        
        def load_sprite(filename):
            path = os.path.join("../assets/images/", filename)
            try:
                img = pygame.image.load(path).convert_alpha()
                return pygame.transform.scale(img, (self.cell_size, self.cell_size))
            except FileNotFoundError:
                print(f"Warning: {path} not found.")
                return None

        textures[Textures.EMPTY] = load_sprite("grass.jpg")
        textures[Textures.RED_APPLE] = load_sprite("red_apple.png")
        textures[Textures.GREEN_APPLE] = load_sprite("green_apple.png")
        textures[Textures.WALL] = load_sprite("wall.png")
        textures[Textures.SNAKE_HEAD_UP] = load_sprite("snake_head_tongue.png")
        textures[Textures.SNAKE_HEAD_DOWN] = pygame.transform.rotate(textures[Textures.SNAKE_HEAD_UP], 180)
        textures[Textures.SNAKE_HEAD_LEFT] = pygame.transform.rotate(textures[Textures.SNAKE_HEAD_UP], 90)
        textures[Textures.SNAKE_HEAD_RIGHT] = pygame.transform.rotate(textures[Textures.SNAKE_HEAD_UP], -90)
        textures[Textures.SNAKE_BODY] = load_sprite("snake_body.png")

        textures[Textures.SNAKE_TAIL_UP] = load_sprite("snake_tail.png")
        textures[Textures.SNAKE_TAIL_DOWN] = pygame.transform.rotate(textures[Textures.SNAKE_TAIL_UP], 180)
        textures[Textures.SNAKE_TAIL_LEFT] = pygame.transform.rotate(textures[Textures.SNAKE_TAIL_UP], 90)
        textures[Textures.SNAKE_TAIL_RIGHT] = pygame.transform.rotate(textures[Textures.SNAKE_TAIL_UP], -90)

        return textures

    def render_frame(self, board_array, head_direction, tail_direction):
        """
        Renders the current board state to the display surface.
        :param board_array: A 1D NumPy array representing the current board state.
        :param head_direction: The direction the snake's head is facing.
        :param tail_direction: The direction the snake's tail is facing.
        """
        self.screen.fill((0, 0, 0))

        cell_size = self.window_size // self.board_size
        for index, value in enumerate(board_array):
            row = index // self.board_size
            col = index % self.board_size
            x = col * cell_size
            y = row * cell_size

            if value == Entity.SNAKE_HEAD:
                texture = self.textures.get(head_direction)    
            elif value == Entity.SNAKE_TAIL:
                texture = self.textures.get(tail_direction)
            else:
                texture = self.textures.get(value)
            
            if texture:
                self.screen.blit(texture, (x, y))
            else:
                pygame.draw.rect(self.screen, (255, 0, 255), (x, y, self.cell_size, self.cell_size))

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
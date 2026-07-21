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

import sys

from constants import AppState, Entity, Textures
from app_config.config import GLOBAL_CONFIG
import numpy as np
import pygame
import os

class GraphicsEngine:
    def __init__(self, window_size: int, board_size: int, fps: int):
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

        self.font = pygame.font.SysFont("Arial", 24, bold=True)

        self.textures = self._load_textures()

    def _load_textures(self):
        textures = {}
        
        def load_sprite(filename: str, full_scale: bool = False):
            path = os.path.join("../assets/images/", filename)
            try:
                img = pygame.image.load(path).convert_alpha()
                if full_scale:
                    print(f"Loaded texture: {filename} at full scale {self.window_size}x{self.window_size} ")
                    return pygame.transform.scale(img, (self.window_size, self.window_size))
                elif not full_scale:
                    return pygame.transform.scale(img, (self.cell_size, self.cell_size))
                else:
                    return img
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
        textures[Textures.LOBBY_BACKGROUND] = load_sprite("lobby_background.png", full_scale=True)
        textures[Textures.SNAKE_TAIL_UP] = load_sprite("snake_tail.png")
        textures[Textures.SNAKE_TAIL_DOWN] = pygame.transform.rotate(textures[Textures.SNAKE_TAIL_UP], 180)
        textures[Textures.SNAKE_TAIL_LEFT] = pygame.transform.rotate(textures[Textures.SNAKE_TAIL_UP], 90)
        textures[Textures.SNAKE_TAIL_RIGHT] = pygame.transform.rotate(textures[Textures.SNAKE_TAIL_UP], -90)

        return textures

    def render_frame(self, board_array: np.ndarray, head_direction: Textures, tail_direction: Textures):
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
    
    def render_lobby(self):
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.window_size, self.window_size))
        
        if self.textures.get(Textures.LOBBY_BACKGROUND):
            self.screen.blit(self.textures[Textures.LOBBY_BACKGROUND], (0, 0))
            
        self.draw_text("Learn2Slither Engine", (0, 255, 100), self.window_size // 2, 100)
        
        center_x = self.window_size // 2 - 175
        play_btn = pygame.Rect(center_x, 200, 350, 50)
        watch_train_btn = pygame.Rect(center_x, 270, 350, 50)
        watch_exploration_train_btn = pygame.Rect(center_x, 340, 350, 50)
        watch_play_exploitation_btn = pygame.Rect(center_x, 410, 350, 50)
        config_btn = pygame.Rect(center_x, 480, 350, 50)
        
        btn_normal = (50, 50, 50, 60)
        btn_hover = (100, 100, 200, 120)
        
        self.draw_button(play_btn, "HUMAN OVERRIDE", btn_normal, btn_hover)
        self.draw_button(watch_train_btn, "TRAINING MODE", btn_normal, btn_hover)
        self.draw_button(watch_exploration_train_btn, "RANDOM EXPLORATION", btn_normal, btn_hover)
        self.draw_button(watch_play_exploitation_btn, "AGENT INFERENCE", btn_normal, btn_hover)
        self.draw_button(config_btn, "ENGINE SETTINGS", btn_normal, btn_hover)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
                
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if play_btn.collidepoint(event.pos):
                    GLOBAL_CONFIG.update('mode', "play")
                    return AppState.PLAYING
                    
                if watch_train_btn.collidepoint(event.pos):
                    GLOBAL_CONFIG.update('mode', "watch-train")
                    return AppState.PLAYING
                    
                if watch_exploration_train_btn.collidepoint(event.pos):
                    GLOBAL_CONFIG.update('mode', "watch-exploration-train")
                    return AppState.PLAYING
                    
                if watch_play_exploitation_btn.collidepoint(event.pos):
                    GLOBAL_CONFIG.update('mode', "watch-exploitation")
                    return AppState.PLAYING
                    
                if config_btn.collidepoint(event.pos):
                    return AppState.CONFIG
                    
        pygame.display.flip()
        return AppState.LOBBY

    def render_config(self):
        return AppState.PLAYING

    def draw_text(self, text: str, color: tuple, x: int, y: int, center: bool = True):
        """
        Renders and blits text to the screen.
        If center=True, x and y represent the center of the text.
        If center=False, x and y represent the top-left corner.
        """
        text_surface = self.font.render(text, True, color)
        text_rect = text_surface.get_rect()
        
        if center:
            text_rect.center = (x, y)
        else:
            text_rect.topleft = (x, y)
            
        self.screen.blit(text_surface, text_rect)
        return text_rect

    def draw_button(self, rect: pygame.Rect, text: str, default_color: tuple, hover_color: tuple, text_color: tuple = (255, 255, 255)):
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = rect.collidepoint(mouse_pos)
        current_color = hover_color if is_hovered else default_color
        
        alpha_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(alpha_surface, current_color, alpha_surface.get_rect())
        
        self.screen.blit(alpha_surface, (rect.x, rect.y))        
        pygame.draw.rect(self.screen, (200, 200, 200), rect, width=2)
        
        self.draw_text(text, text_color, rect.centerx, rect.centery, center=True)
        return is_hovered
        
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
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

from moves_engine.move_engine import MovesEngine

class GraphicsEngine:
    def __init__(self, window_size: int):
        """
        Initializes the graphics engine with the specified window size, board size, and frames per second (fps).
        :param window_size: The size of the window in pixels.
        :param board_size: The size of the board (number of cells in one dimension).
        :param fps: The frames per second for rendering.
        """
        pygame.init()
        self.screen = pygame.display.set_mode((window_size, window_size), pygame.HIDDEN)
        self.clock = pygame.time.Clock()
        self.window_size = window_size
        self.font = pygame.font.SysFont("Arial", 24, bold=True)
        self.textures = self._load_textures()
        self.screen = pygame.display.set_mode((window_size, window_size))

    @property
    def cell_size(self):
        """
        Dynamically calculates the square cell size based on the live config.
        Returns a single integer.
        """
        board_size: int = GLOBAL_CONFIG.get('board_size')
        return self.window_size // board_size
    
    def _load_textures(self):
        textures = {}
        
        def load_sprite(filename: str, full_scale: bool = False, darken_factor: int = 255):
            path = os.path.join("../assets/images/", filename)
            try:
                img = pygame.image.load(path).convert_alpha()
                
                if darken_factor < 255:
                    dark_color = (darken_factor, darken_factor, darken_factor, 255)
                    img.fill(dark_color, special_flags=pygame.BLEND_RGBA_MULT)

                if full_scale:
                    return pygame.transform.scale(img, (self.window_size, self.window_size))
                else:
                    return pygame.transform.scale(img, (self.cell_size, self.cell_size))
            except FileNotFoundError:
                print(f"Warning: {path} not found.")
                sys.exit(1)

        textures[Textures.LOBBY_BACKGROUND] = load_sprite("lobby_background.png", full_scale=True)
        textures[Textures.GAME_BACKGROUND] = load_sprite("game_background.jpg", full_scale=True, darken_factor=160)
        textures[Textures.GREEN_APPLE] = load_sprite("green_apple.png")
        textures[Textures.SNAKE_BODY] = load_sprite("snake_body.png")
        textures[Textures.RED_APPLE] = load_sprite("red_apple.png")
        textures[Textures.WALL] = load_sprite("wall.png")

        base_head_up = load_sprite("snake_head.png")
        tongue_head_up = load_sprite("snake_head_tongue.png")
        textures[Textures.SNAKE_HEAD_UP] = [base_head_up, tongue_head_up]
        textures[Textures.SNAKE_HEAD_DOWN] = [
            pygame.transform.rotate(base_head_up, 180),
            pygame.transform.rotate(tongue_head_up, 180)
        ]
        textures[Textures.SNAKE_HEAD_LEFT] = [
            pygame.transform.rotate(base_head_up, 90),
            pygame.transform.rotate(tongue_head_up, 90)
        ]
        textures[Textures.SNAKE_HEAD_RIGHT] = [
            pygame.transform.rotate(base_head_up, -90),
            pygame.transform.rotate(tongue_head_up, -90)
        ]

        textures[Textures.SNAKE_TAIL_UP] = load_sprite("snake_tail.png")
        textures[Textures.SNAKE_TAIL_LEFT] = pygame.transform.rotate(textures[Textures.SNAKE_TAIL_UP], 90)
        textures[Textures.SNAKE_TAIL_RIGHT] = pygame.transform.rotate(textures[Textures.SNAKE_TAIL_UP], -90)
        textures[Textures.SNAKE_TAIL_DOWN] = pygame.transform.rotate(textures[Textures.SNAKE_TAIL_UP], 180)
        
        return textures

    def refresh_textures(self):
        """
        Forces the engine to reload and rescale all image assets 
        """
        self.textures = self._load_textures()

    def render_live_results_statistics(
            self,
            score: int = 0,
            green_apple_eaten: int = 0,
            red_apple_eaten: int = 0
        ):
        """
        Renders a small HUD panel with score, green/red apples eaten, and live FPS,
        with values aligned in a column regardless of label length.
        """
        live_fps = int(self.clock.get_fps())

        padding = 14
        line_gap = 8
        value_gap = 14
        line_height = self.font.get_height() + line_gap

        rows = [
            ("Score:", str(score), (255, 255, 255)),
            ("Green Apples:", str(green_apple_eaten), (80, 255, 120)),
            ("Red Apples:", str(red_apple_eaten), (255, 90, 90)),
            ("FPS:", str(live_fps), (170, 170, 180)),
        ]

        dot_col_width = 16
        label_col_width = max(self.font.size(label)[0] for label, _, _ in rows)
        value_col_x = padding + dot_col_width + label_col_width + value_gap

        hud_width = value_col_x + max(self.font.size(value)[0] for _, value, _ in rows) + padding
        hud_height = line_height * len(rows) - line_gap + padding * 2

        hud_surface = pygame.Surface((hud_width, hud_height), pygame.SRCALPHA)
        pygame.draw.rect(hud_surface, (15, 15, 20, 150), hud_surface.get_rect(), border_radius=10)
        pygame.draw.rect(hud_surface, (255, 255, 255, 40), hud_surface.get_rect(), width=1, border_radius=10)

        for i, (label, value, color) in enumerate(rows):
            row_y = padding + i * line_height

            dot_center = (padding + 4, row_y + self.font.get_height() // 2)
            pygame.draw.circle(hud_surface, color, dot_center, 4)

            label_surf = self.font.render(label, True, color)
            hud_surface.blit(label_surf, (padding + dot_col_width, row_y))

            value_surf = self.font.render(value, True, color)
            hud_surface.blit(value_surf, (value_col_x, row_y))

        self.screen.blit(hud_surface, (10, 10))


    def render_game_results_statistics(
            self,
            score: int = 0,
            green_apple_eaten: int = 0,
            red_apple_eaten: int = 0
        ):
        """
        Renders the after-game results screen inside a centered card, with a
        "Return to Lobby" button. Blocks the game loop until the user clicks
        the button or presses Esc.
        """
        overlay = pygame.Surface((self.window_size, self.window_size), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 190))
        self.screen.blit(overlay, (0, 0))

        center_x = self.window_size // 2
        center_y = self.window_size // 2

        title_surf = self.font.render("SIMULATION TERMINATED", True, (255, 70, 70))
        scale = 1.6
        title_surf = pygame.transform.smoothscale(
            title_surf,
            (int(title_surf.get_width() * scale), int(title_surf.get_height() * scale))
        )

        rows = [
            ("Final Score:", str(score), (255, 255, 255)),
            ("Green Apples Collected:", str(green_apple_eaten), (80, 255, 120)),
            ("Red Apples Penalties:", str(red_apple_eaten), (255, 100, 100)),
        ]
        label_col_width = max(self.font.size(label)[0] for label, _, _ in rows)
        value_gap = 16
        row_height = self.font.get_height()
        line_gap = 12
        section_gap = 30
        btn_width, btn_height = 260, 54

        stats_block_width = label_col_width + value_gap + max(self.font.size(v)[0] for _, v, _ in rows)
        content_height = (
            title_surf.get_height() + section_gap
            + row_height * len(rows) + line_gap * (len(rows) - 1)
            + section_gap + btn_height
        )
        content_width = max(title_surf.get_width(), stats_block_width, btn_width)

        card_padding = 40
        card_rect = pygame.Rect(0, 0, content_width + card_padding * 2, content_height + card_padding * 2)
        card_rect.center = (center_x, center_y)

        card_surface = pygame.Surface(card_rect.size, pygame.SRCALPHA)
        pygame.draw.rect(card_surface, (25, 25, 30, 230), card_surface.get_rect(), border_radius=16)
        pygame.draw.rect(card_surface, (255, 255, 255, 35), card_surface.get_rect(), width=1, border_radius=16)
        self.screen.blit(card_surface, card_rect.topleft)

        y = card_rect.top + card_padding
        self.screen.blit(title_surf, title_surf.get_rect(midtop=(center_x, y)))
        y += title_surf.get_height() + section_gap

        # left-align the whole label/value block, but center that block within the card
        block_left = center_x - stats_block_width // 2
        for label, value, color in rows:
            label_surf = self.font.render(label, True, color)
            value_surf = self.font.render(value, True, color)
            self.screen.blit(label_surf, (block_left, y))
            self.screen.blit(value_surf, (block_left + label_col_width + value_gap, y))
            y += row_height + line_gap
        y += section_gap - line_gap

        btn_rect = pygame.Rect(0, 0, btn_width, btn_height)
        btn_rect.midtop = (center_x, y)
        btn_normal = (50, 50, 55)
        btn_hover = (90, 100, 200)

        waiting = True
        while waiting:
            mouse_pos = pygame.mouse.get_pos()
            hovering = btn_rect.collidepoint(mouse_pos)

            pygame.draw.rect(self.screen, btn_hover if hovering else btn_normal, btn_rect, border_radius=8)
            pygame.draw.rect(self.screen, (255, 255, 255), btn_rect, width=2, border_radius=8)

            btn_text = self.font.render("RETURN TO LOBBY", True, (255, 255, 255))
            self.screen.blit(btn_text, btn_text.get_rect(center=btn_rect.center))

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    if hovering:
                        return AppState.LOBBY
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return AppState.LOBBY

            pygame.display.flip()
            self.clock.tick(GLOBAL_CONFIG.get('fps_cap'))

    def render_user_game_frame(
            self,
            board_array: np.ndarray,
            head_direction: Textures,
            tail_direction: Textures,
            is_dead: bool,
            current_state: AppState,
            score: int = 0,
            green_apple_eaten: int = 0,
            red_apple_eaten: int = 0
        ):
        """
        Renders the current board state to the display surface.
        :param board_array: A 1D NumPy array representing the current board state.
        :param head_direction: The direction the snake's head is facing.
        :param tail_direction: The direction the snake's tail is facing.
        :param is_dead: A boolean indicating whether the snake is dead.
        :param current_state: The current state of the application.
        """
        if is_dead:
            return self.render_game_results_statistics(
                score=score,
                green_apple_eaten=green_apple_eaten,
                red_apple_eaten=red_apple_eaten
            )
        else:
            self.screen.fill((0, 0, 0))
            if self.textures.get(Textures.GAME_BACKGROUND):
                self.screen.blit(self.textures[Textures.GAME_BACKGROUND], (0, 0))


            current_time = pygame.time.get_ticks()
            animation_cycle = current_time % 1000
            frame_index = 1 if animation_cycle > 600 else 0
            for index, value in enumerate(board_array):
                row = index // GLOBAL_CONFIG.get('board_size')
                col = index % GLOBAL_CONFIG.get('board_size')
                x = col * self.cell_size
                y = row * self.cell_size

                if value == Entity.SNAKE_HEAD:
                    texture = self.textures.get(head_direction)[frame_index]
                elif value == Entity.SNAKE_TAIL:
                    texture = self.textures.get(tail_direction)
                else:
                    texture = self.textures.get(value)
                
                if value != Entity.EMPTY:
                    if texture:
                        self.screen.blit(texture, (x, y))
                    else:
                        pygame.draw.rect(self.screen, (255, 0, 255), (x, y, self.cell_size, self.cell_size))

            self.render_live_results_statistics(
                score=score,
                green_apple_eaten=green_apple_eaten,
                red_apple_eaten=red_apple_eaten
            )
            pygame.display.flip()
            self.clock.tick(GLOBAL_CONFIG.get('fps_cap'))

        return current_state

    def render_agent_train(self):
        pass

    def render_lobby(self):
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.window_size, self.window_size))
        if self.textures.get(Textures.LOBBY_BACKGROUND):
            self.screen.blit(self.textures[Textures.LOBBY_BACKGROUND], (0, 0))
            
        self.draw_text("Learn2Slither Engine", (0, 255, 100), self.window_size // 2, 100)
        
        center_x = self.window_size // 2 - 175
        train_btn = pygame.Rect(center_x, 200, 350, 50)
        play_btn = pygame.Rect(center_x, 270, 350, 50)
        watch_train_btn = pygame.Rect(center_x, 340, 350, 50)
        watch_exploration_train_btn = pygame.Rect(center_x, 410, 350, 50)
        watch_play_exploitation_btn = pygame.Rect(center_x, 480, 350, 50)
        config_btn = pygame.Rect(center_x, 550, 350, 50)
        
        btn_normal = (50, 50, 50, 60)
        btn_hover = (100, 100, 200, 120)
        
        self.draw_button(train_btn, "TRAIN AGENT", btn_normal, btn_hover)
        self.draw_button(play_btn, "HUMAN OVERRIDE", btn_normal, btn_hover)
        self.draw_button(watch_train_btn, "WATCH TRAINING", btn_normal, btn_hover)
        self.draw_button(watch_exploration_train_btn, "RANDOM EXPLORATION", btn_normal, btn_hover)
        self.draw_button(watch_play_exploitation_btn, "AGENT INFERENCE", btn_normal, btn_hover)
        self.draw_button(config_btn, "ENGINE SETTINGS", btn_normal, btn_hover)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                sys.exit()
                
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if watch_train_btn.collidepoint(event.pos):
                    GLOBAL_CONFIG.update('mode', "train")
                    return AppState.TRAIN

                if play_btn.collidepoint(event.pos):
                    GLOBAL_CONFIG.update('mode', "play")
                    return AppState.PLAYING

                if watch_train_btn.collidepoint(event.pos):
                    GLOBAL_CONFIG.update('mode', "watch-train")
                    return AppState.WATCH_TRAIN

                if watch_exploration_train_btn.collidepoint(event.pos):
                    GLOBAL_CONFIG.update('mode', "watch-exploration-train")
                    return AppState.WATCH_EXPLORATION_TRAIN

                if watch_play_exploitation_btn.collidepoint(event.pos):
                    GLOBAL_CONFIG.update('mode', "watch-exploitation")
                    return AppState.WATCH_EXPLOITATION

                if config_btn.collidepoint(event.pos):
                    return AppState.CONFIG

        pygame.display.flip()
        return AppState.LOBBY

    def render_config(self):
        """
        Renders the configuration screen where users can view and modify engine settings.
        :return: The next AppState based on user interaction.
        """        
        pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, self.window_size, self.window_size))
        if self.textures.get(Textures.LOBBY_BACKGROUND):
            self.screen.blit(self.textures[Textures.LOBBY_BACKGROUND], (0, 0))

        if not hasattr(self, 'active_input'):
            self.active_input = None
            self.input_buffer = ""
            self.error_msg = ""
            self.error_time = 0

        self.draw_text_aligned("ENGINE CONFIGURATION MATRIX", (0, 255, 100), self.window_size // 2, self.window_size // 8, align="center")

        col_width = 400
        btn_height = 40
        padding = 10
        start_y = self.window_size // 4
        
        center_x = (self.window_size - col_width) // 2
        
        config_keys = [
            ("board_size", "Board Size"),
            ("fps_cap", "Target FPS"),
            ("reward_green_apple", "Reward: Green Apple"),
            ("reward_red_apple", "Reward: Red Apple"),
            ("reward_death", "Reward: Death"),
            ("reward_step", "Reward: Step")
        ]

        btn_normal = (50, 50, 50, 60)
        btn_hover = (100, 100, 200, 120)
        interactive_zones = []

        self.draw_text_aligned("ENVIRONMENT RULES", (200, 200, 200), self.window_size // 2, start_y - 25, align="center")
        for i, (key, label) in enumerate(config_keys):
            rect = pygame.Rect(center_x, start_y + (i * (btn_height + padding)), col_width, btn_height)
            is_active = (self.active_input == key)
            display_val = self.input_buffer if is_active else getattr(GLOBAL_CONFIG, key)
            
            self.draw_text_input_row(rect, label, display_val, is_active, btn_normal, btn_hover)
            interactive_zones.append((rect, key))

        if self.error_msg and pygame.time.get_ticks() - self.error_time < 3000:
            self.draw_text_aligned(self.error_msg, (255, 50, 50), self.window_size // 2, self.window_size - 110, align="center")

        back_btn_rect = pygame.Rect(self.window_size // 2 - 150, self.window_size - 80, 300, 50)
        self.draw_button(back_btn_rect, "RETURN TO LOBBY", btn_normal, btn_hover)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.active_input:
                    try:
                        GLOBAL_CONFIG.update(self.active_input, self.input_buffer)
                        self.error_msg = ""
                    except ValueError as e:
                        self.error_msg = str(e)
                        self.error_time = pygame.time.get_ticks()
                    self.active_input = None

                if back_btn_rect.collidepoint(event.pos):
                    return AppState.LOBBY

                for rect, key in interactive_zones:
                    if rect.collidepoint(event.pos):
                        self.active_input = key
                        self.input_buffer = str(getattr(GLOBAL_CONFIG, key))
                        if self.input_buffer == "None":
                            self.input_buffer = ""
                        break
                        
            elif event.type == pygame.KEYDOWN and self.active_input:
                if event.key == pygame.K_RETURN:
                    try:
                        GLOBAL_CONFIG.update(self.active_input, self.input_buffer)
                        self.active_input = None
                        self.error_msg = ""
                    except ValueError as e:
                        self.error_msg = str(e)
                        self.error_time = pygame.time.get_ticks()
                
                elif event.key == pygame.K_BACKSPACE:
                    self.input_buffer = self.input_buffer[:-1]
                
                elif event.key == pygame.K_ESCAPE:
                    self.active_input = None
                
                else:
                    self.input_buffer += event.unicode

        pygame.display.flip()
        self.clock.tick(GLOBAL_CONFIG.get('fps_cap'))

        return AppState.CONFIG

    def draw_text_aligned(self, text, color, x, y, align="center"):
        """Utility to draw text anchored left, right, or center."""
        text_surface = self.font.render(str(text), True, color)
        text_rect = text_surface.get_rect()
        
        if align == "center":
            text_rect.center = (x, y)
        elif align == "left":
            text_rect.midleft = (x, y)
        elif align == "right":
            text_rect.midright = (x, y)
            
        self.screen.blit(text_surface, text_rect)
        return text_rect

    def draw_text_input_row(self, rect, label, display_value, is_active, default_color, hover_color, active_color=(20, 80, 40, 220)):
        """Draws a configuration row that supports text entry state."""
        mouse_pos = pygame.mouse.get_pos()
        is_hovered = rect.collidepoint(mouse_pos)
        
        current_color = active_color if is_active else (hover_color if is_hovered else default_color)
        
        alpha_surface = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        pygame.draw.rect(alpha_surface, current_color, alpha_surface.get_rect())
        self.screen.blit(alpha_surface, (rect.x, rect.y))
        
        border_color = (0, 255, 100) if is_active else (100, 100, 100)
        pygame.draw.rect(self.screen, border_color, rect, width=2 if is_active else 1)
        
        self.draw_text_aligned(label, (255, 255, 255), rect.x + 15, rect.centery, align="left")
        
        val_color = (255, 255, 255) if is_active else (0, 255, 100)
        text_to_draw = str(display_value)
        
        if is_active and (pygame.time.get_ticks() % 1000 < 500):
            text_to_draw += "|"
            
        self.draw_text_aligned(text_to_draw, val_color, rect.right - 15, rect.centery, align="right")
        
        return is_hovered

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
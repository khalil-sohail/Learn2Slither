from collections import deque
from app_config.config import GLOBAL_CONFIG
from constants import Entity

import numpy as np

VISION_DIRECTIONS = (
    (-1, 0),
    (0, 1),
    (1, 0),
    (0, -1),
)

VISION_BODY_ENTITIES = frozenset({
    Entity.SNAKE_BODY,
    Entity.SNAKE_TAIL_UP,
    Entity.SNAKE_TAIL_DOWN,
    Entity.SNAKE_TAIL_LEFT,
    Entity.SNAKE_TAIL_RIGHT,
})

class BoardState:
    """"""
    def __init__(self):
        """"""
        self.board_size = GLOBAL_CONFIG.get("board_size")
        self.print_vision = GLOBAL_CONFIG.get("print_vision")
        self.board = np.zeros((self.board_size * self.board_size), dtype=int)
        self.body = deque()
        self.red_apple = deque()
        self.green_apple = deque()
        self.red_apple_eaten = 0
        self.green_apple_eaten = 0
        self.board_score = 0
        self.head_index: int = 0

    def reset_game(self):
        """"""
        self.board_size = GLOBAL_CONFIG.get("board_size")
        self.print_vision = GLOBAL_CONFIG.get("print_vision")
        self.board = np.zeros((self.board_size * self.board_size), dtype=int)
        self.body.clear()
        self.red_apple.clear()
        self.green_apple.clear()
        self.red_apple_eaten = 0
        self.green_apple_eaten = 0
        self.board_score = 0

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


    

    def vision_symbol(self, cell) -> str:
        """Convert a board entity into its visible symbol."""
        if cell == Entity.WALL:
            return "W"
        if cell in VISION_BODY_ENTITIES:
            return "S"
        if cell == Entity.GREEN_APPLE:
            return "G"
        if cell == Entity.RED_APPLE:
            return "R"
        return "0"

    def scan_vision_ray(self, head_row: int, head_col: int, dr: int, dc: int) -> list[str]:
        """Scan one direction from the head until a wall is reached."""
        row, col = head_row, head_col
        ray = []

        while True:
            row += dr
            col += dc

            if not (
                0 <= row < self.board_size
                and 0 <= col < self.board_size
            ):
                raise RuntimeError(
                    "Vision ray reached the board boundary "
                    "without encountering a wall."
                )

            cell = self.board[row * self.board_size + col]
            symbol = self.vision_symbol(cell)
            ray.append(symbol)

            if symbol == "W":
                return ray

    def get_visible_rays(self) -> list[list[str]]:
        """Return the four visible rays in N, E, S, W order."""
        head_row, head_col = divmod(
            self.head_index,
            self.board_size,
        )

        return [
            self.scan_vision_ray(head_row, head_col, dr, dc)
            for dr, dc in VISION_DIRECTIONS
        ]

    def normalize_vision_distance(self, distance: int, dimension: int) -> float:
        """Normalize distance: closer = higher, absent = 0."""
        return 1.0 - (distance - 1) / (dimension - 1)

    def get_ray_distance_features(self, ray: list[str], dimension: int) -> list[float]:
        """
        Return five normalized distances:
          [0] Wall
          [1] First body segment
          [2] Second body segment
          [3] First green apple
          [4] First red apple

        Missing objects have distance 0.0.
        """
        body_positions = [
            i + 1
            for i, symbol in enumerate(ray)
            if symbol == "S"
        ]

        def first_distance(symbol: str) -> int:
            return ray.index(symbol) + 1 if symbol in ray else 0

        distances = [
            first_distance("W"),
            body_positions[0] if len(body_positions) > 0 else 0,
            body_positions[1] if len(body_positions) > 1 else 0,
            first_distance("G"),
            first_distance("R"),
        ]

        return [
            self.normalize_vision_distance(distance, dimension)
            if distance else 0.0
            for distance in distances
        ]

    def get_ray_corridor_length(self, ray: list[str], dimension: int) -> float:
        """
        Normalized number of consecutive traversable cells
        before encountering a wall or body.

        0.0 means the next cell is blocked.
        """
        free_cells = 0

        for symbol in ray:
            if symbol in ("W", "S"):
                break
            free_cells += 1

        return free_cells / max(1, dimension - 3)


    def get_clear_green_proximity(self, ray: list[str], dimension: int) -> float:
        """
        Return the normalized proximity of the first green apple
        reachable along the straight ray without crossing a body.

        Returns 0.0 when no unobstructed green apple is visible.
        """
        for distance, symbol in enumerate(ray, start=1):
            if symbol in ("W", "S"):
                break

            if symbol == "G":
                return self.normalize_vision_distance(
                    distance, dimension
                )

        return 0.0

    def get_vision_density_features(self, rays: list[list[str]]) -> list[float]:
        """Return the visible body density for each direction."""
        return [
            ray.count("S") / len(ray)
            for ray in rays
        ]

    def normalize_board_size(self, size: int) -> float:
        """
        Normalize board size without imposing a maximum size.

        Returns values in (0, 1) for positive board sizes.
        """
        return size / (size + 20.0)

    def print_visible_rays(self, rays: list[list[str]]) -> None:
        """Display the four-direction vision in the terminal."""
        north, east, south, west = rays

        left = "".join(reversed(west))
        right = "".join(east)
        padding = " " * len(left)

        print()

        for symbol in reversed(north):
            print(padding + symbol)

        print(left + "H" + right)

        for symbol in south:
            print(padding + symbol)

        print()

    def get_snake_vision_vector(self) -> np.ndarray:
        """
        Build a 33-feature observation using only visible cells.

        Each direction has eight features:
          [0] Wall proximity
          [1] First body proximity
          [2] Second body proximity
          [3] First green apple proximity
          [4] First red apple proximity
          [5] Clear corridor length
          [6] Unobstructed green apple proximity
          [7] Visible body density

        Layout:
          [0:8]   North
          [8:16]  East
          [16:24] South
          [24:32] West
          [32]    Normalized board size

        All features are derived exclusively from the four visible rays.
        """
        rays = self.get_visible_rays()

        height = len(rays[0]) + len(rays[2]) + 1
        width = len(rays[1]) + len(rays[3]) + 1

        assert width == height
        size = width

        body_densities = self.get_vision_density_features(rays)
        features = []
        for ray, body_density in zip(rays, body_densities):
            features.extend(self.get_ray_distance_features(ray, size))
            features.append(self.get_ray_corridor_length(ray, size))
            features.append(self.get_clear_green_proximity(ray, size))
            features.append(body_density)

        features.append(self.normalize_board_size(size))

        if self.print_vision:
            self.print_visible_rays(rays)

        assert len(features) == 33
        return np.asarray(features, dtype=np.float32)

























































    # def get_snake_vision_vector(self):
    #     """
    #     Calculates and returns the 29-value float32 NumPy array representing what
    #     the snake sees in its current state.

    #     Layout:
    #         [0:16]  4 features per direction (N,E,S,W order): wall/position,
    #                 body distance, green apple distance, red apple distance
    #                 (same normalization scheme as before)
    #         [16:20] one-hot current heading (agent state, not board info) #####this feature must be removed, violates the subject rules.#####
    #         [20:23] local density features over all visible (plus-shape) cells:
    #                 /*local_body_density, local_free_density*/, local_wall_density,
    #                 local_green_density, local_red_density
    #         [23:27] immediate danger flags (N,E,S,W order): 1.0 if the very next
    #                 cell in that direction is a wall or body segment, else 0.0
    #         [27]    normalized board width (board_width / 100)
    #         [28]    normalized board height (board_height / 100)
    #     :return: A 29-value float32 NumPy array.
    #     """
    #     vision_vector = np.zeros(29, dtype=np.float32)
    #     directions = [
    #         (-1, 0),    # North
    #         (0, 1),     # East
    #         (1, 0),     # South
    #         (0, -1)     # West
    #     ]
    #     head_row = self.head_index // self.board_size
    #     head_col = self.head_index % self.board_size

    #     rays_chars = {
    #         (-1, 0): [],
    #         (0, 1): [],
    #         (1, 0): [],
    #         (0, -1): []
    #     }

    #     BODY = [
    #         Entity.SNAKE_BODY,
    #         Entity.SNAKE_TAIL_UP,
    #         Entity.SNAKE_TAIL_DOWN,
    #         Entity.SNAKE_TAIL_LEFT,
    #         Entity.SNAKE_TAIL_RIGHT
    #     ]

    #     body_count = 0
    #     empty_count = 0
    #     wall_count = 0
    #     green_count = 0
    #     red_count = 0
    #     danger = [0.0, 0.0, 0.0, 0.0]   # N, E, S, W

    #     for i, (dr, dc) in enumerate(directions):
    #         distance = 0
    #         body_distance = 0
    #         green_distance = 0
    #         red_distance = 0

    #         row, col = head_row, head_col
    #         while True:
    #             row += dr
    #             col += dc

    #             if (row < 0 or row >= self.board_size or col < 0 or col >= self.board_size):
    #                 break

    #             distance += 1

    #             index = row * self.board_size + col
    #             cell = self.board[index]

    #             if distance == 1:
    #                 danger[i] = 1.0 if (cell == Entity.WALL or cell in BODY) else 0.0

    #             if cell == Entity.WALL:
    #                 rays_chars[(dr, dc)].append('W')
    #                 wall_count += 1
    #                 break
    #             elif cell in BODY:
    #                 rays_chars[(dr, dc)].append('S')
    #                 body_count += 1
    #                 if body_distance == 0: body_distance = distance
    #             elif cell == Entity.GREEN_APPLE:
    #                 rays_chars[(dr, dc)].append('G')
    #                 green_count += 1
    #                 if green_distance == 0: green_distance = distance
    #             elif cell == Entity.RED_APPLE:
    #                 rays_chars[(dr, dc)].append('R')
    #                 red_count += 1
    #                 if red_distance == 0: red_distance = distance
    #             else:
    #                 rays_chars[(dr, dc)].append('0')
    #                 empty_count += 1

    #         vision_vector[i*4] = distance
    #         vision_vector[i*4+1] = body_distance if body_distance else 0.0
    #         vision_vector[i*4+2] = green_distance if green_distance else 0.0
    #         vision_vector[i*4+3] = red_distance if red_distance else 0.0

    #     north, east, south, west = vision_vector[0], vision_vector[4], vision_vector[8], vision_vector[12]
    #     board_height = north + south + 1
    #     board_width  = east + west + 1

    #     for i in range(4):
    #         dimension = board_height if i in (0, 2) else board_width  # N,S: height, E,W: width
    #         for j in range(4):
    #             idx = i * 4 + j
    #             value = vision_vector[idx]
    #             if value == 0.0:
    #                 continue
    #             vision_vector[idx] = 1.0 - (value - 1) / (dimension - 1)
    #             # vision_vector[idx] = (dimension - value) / (dimension - 1)

    #     heading = np.zeros(4, dtype=np.float32)
    #     head = self.board[self.head_index]
    #     if head == Entity.SNAKE_HEAD_UP:
    #         heading[0] = 1.0
    #     elif head == Entity.SNAKE_HEAD_RIGHT:
    #         heading[1] = 1.0
    #     elif head == Entity.SNAKE_HEAD_DOWN:
    #         heading[2] = 1.0
    #     elif head == Entity.SNAKE_HEAD_LEFT:
    #         heading[3] = 1.0
    #     vision_vector[16:20] = heading

    #     visible_cells = body_count + empty_count + wall_count + green_count + red_count
    #     if visible_cells == 0:
    #         # Shouldn't happen (every ray scans at least its own wall cell),
    #         # guard anyway rather than divide by zero.
    #         # local_body_density = local_free_density = 0.0
    #         local_wall_density = 0.0
    #         local_green_density = local_red_density = 0.0
    #     else:
    #         # local_body_density  = body_count  / visible_cells
    #         # local_free_density  = empty_count / visible_cells
    #         local_wall_density  = wall_count  / visible_cells
    #         local_green_density = green_count / visible_cells
    #         local_red_density   = red_count   / visible_cells

    #     # vision_vector[20] = local_body_density
    #     # vision_vector[21] = local_free_density
    #     vision_vector[20] = local_wall_density
    #     vision_vector[21] = local_green_density
    #     vision_vector[22] = local_red_density

    #     vision_vector[23:27] = danger
    #     vision_vector[27] = board_width / 100
    #     vision_vector[28] = board_height / 100

    #     if self.print_vision:
    #         west_str = "".join(reversed(rays_chars[(0, -1)]))
    #         east_str = "".join(rays_chars[(0, 1)])
    #         padding = " " * len(west_str)
    #         print()
    #         for char in reversed(rays_chars[(-1, 0)]):
    #             print(padding + char)
    #         print(west_str + "H" + east_str)
    #         for char in rays_chars[(1, 0)]:
    #             print(padding + char)
    #         print()

    #     return vision_vector

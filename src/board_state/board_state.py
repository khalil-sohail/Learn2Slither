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
        self.pending_apples = []

    def reset_game(self):
        """
        Reset the board with a randomly positioned, contiguous
        3-cell snake, two green apples, and one red apple.
        """
        self.board_size = GLOBAL_CONFIG.get("board_size")
        self.print_vision = GLOBAL_CONFIG.get("print_vision")

        self.board = np.full(
            self.board_size * self.board_size,
            Entity.EMPTY,
            dtype=int,
        )

        self.body.clear()
        self.red_apple.clear()
        self.green_apple.clear()
        self.pending_apples.clear()

        self.red_apple_eaten = 0
        self.green_apple_eaten = 0
        self.board_score = 0

        self.board[:self.board_size] = Entity.WALL
        self.board[-self.board_size:] = Entity.WALL
        self.board[::self.board_size] = Entity.WALL
        self.board[self.board_size - 1::self.board_size] = Entity.WALL

        while True:
            head_row = int(np.random.randint(1, self.board_size - 1))
            head_col = int(np.random.randint(1, self.board_size - 1))

            direction = int(np.random.randint(len(VISION_DIRECTIONS)))
            dr, dc = VISION_DIRECTIONS[direction]

            middle_row = head_row - dr
            middle_col = head_col - dc
            tail_row = head_row - 2 * dr
            tail_col = head_col - 2 * dc

            if (1 <= tail_row < self.board_size - 1 and 1 <= tail_col < self.board_size - 1):
                break

        tail_index = tail_row * self.board_size + tail_col
        middle_index = middle_row * self.board_size + middle_col
        head_index = head_row * self.board_size + head_col

        self.body.extend([tail_index, middle_index, head_index])
        self.head_index = head_index


        self.board[head_index] = self.get_head_direction(middle_index, head_index)
        self.board[middle_index] = Entity.SNAKE_BODY
        self.board[tail_index] = self.get_tail_direction(tail_index, middle_index)

        self.spawn_apple(Entity.RED_APPLE)
        self.spawn_apple(Entity.GREEN_APPLE)
        self.spawn_apple(Entity.GREEN_APPLE)

    def move_snake(self, next_index: int):
        """
        Move the snake to a legal destination.
        Returns the entity consumed at next_index.
        Assumes wall and body collisions have already been checked by the caller.
        """
        if not self.body:
            raise RuntimeError("Cannot move a dead snake.")

        ate_apple = self.board[next_index]
        self.board[self.body[-1]] = Entity.SNAKE_BODY

        self.body.append(next_index)
        self.board[next_index] = self.get_head_direction(self.head_index, next_index)

        tail_moves = 1
        apple = Entity.EMPTY
        if ate_apple == Entity.RED_APPLE:
            self.red_apple.remove(next_index)
            self.red_apple_eaten += 1
            self.board_score = max(0, self.board_score - 1)

            apple = Entity.RED_APPLE
            tail_moves = 2
        elif ate_apple == Entity.GREEN_APPLE:
            self.green_apple.remove(next_index)
            self.green_apple_eaten += 1
            self.board_score += 1

            apple = Entity.GREEN_APPLE
            tail_moves = 0

        for _ in range(tail_moves):
            removed_index = self.body.popleft()
            if removed_index != next_index:
                self.board[removed_index] = Entity.EMPTY

        self.head_index = next_index
        if not self.body:
            self.board[next_index] = Entity.EMPTY
            return ate_apple

        if len(self.body) >= 2:
            self.board[self.body[0]] = self.get_tail_direction(self.body[0], self.body[1])

        if apple != Entity.EMPTY:
            self.pending_apples.append(apple)

        self.respawn_pending_apples()
        return ate_apple


    def spawn_apple(self, color: Entity) -> bool:
        empty_indices = np.flatnonzero(self.board == Entity.EMPTY)

        if empty_indices.size == 0:
            return False

        random_index = int(np.random.choice(empty_indices))

        self.board[random_index] = color

        if color == Entity.RED_APPLE:
            self.red_apple.append(random_index)
        elif color == Entity.GREEN_APPLE:
            self.green_apple.append(random_index)

        return True

    def respawn_pending_apples(self):
        remaining = []

        for apple in self.pending_apples:
            if not self.spawn_apple(apple):
                remaining.append(apple)

        self.pending_apples = remaining
        
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
















































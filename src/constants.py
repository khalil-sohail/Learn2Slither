from enum import IntEnum

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Textures(IntEnum):
    SNAKE_HEAD_UP = 0
    SNAKE_HEAD_DOWN = 1
    SNAKE_HEAD_LEFT = 2
    SNAKE_HEAD_RIGHT = 3
    SNAKE_BODY = 4
    SNAKE_TAIL_UP = 5
    SNAKE_TAIL_DOWN = 6
    SNAKE_TAIL_LEFT = 7
    SNAKE_TAIL_RIGHT = 8
    EMPTY = 9
    RED_APPLE = 10
    GREEN_APPLE = 11
    WALL = 12
    LOBBY_BACKGROUND = 13
    GAME_BACKGROUND = 14

class Entity(IntEnum):
    EMPTY = Textures.EMPTY
    SNAKE_HEAD = Textures.SNAKE_HEAD_UP or Textures.SNAKE_HEAD_DOWN or Textures.SNAKE_HEAD_LEFT or Textures.SNAKE_HEAD_RIGHT
    SNAKE_BODY = Textures.SNAKE_BODY
    SNAKE_TAIL = Textures.SNAKE_TAIL_UP or Textures.SNAKE_TAIL_DOWN or Textures.SNAKE_TAIL_LEFT or Textures.SNAKE_TAIL_RIGHT
    RED_APPLE = Textures.RED_APPLE
    GREEN_APPLE = Textures.GREEN_APPLE
    WALL = Textures.WALL

class AppState(IntEnum):
    LOBBY = 0
    CONFIG = 1
    PLAYING = 2
    EXIT = 3
    TRAIN = 4
    WATCH_TRAIN = 5
    WATCH_EXPLORATION_TRAIN = 6
    WATCH_EXPLOITATION = 7


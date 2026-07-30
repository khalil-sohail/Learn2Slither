from enum import IntEnum

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class Entity(IntEnum):
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

class AppState(IntEnum):
    LOBBY = 0
    CONFIG = 1
    EVALUATE = 2
    TRAIN = 3
    HUMAN_PLAYING = 4
    WATCH_TRAIN = 5
    AGENT_PLAYING_EX = 6
    AGENT_PLAYING_EI = 7
    EXIT = 8


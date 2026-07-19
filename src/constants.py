from enum import IntEnum

class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    
class Entity(IntEnum):
    EMPTY = 0
    SNAKE_BODY = 1
    SNAKE_HEAD = 2
    RED_APPLE = 3
    GREEN_APPLE = 4
    WALL = 5

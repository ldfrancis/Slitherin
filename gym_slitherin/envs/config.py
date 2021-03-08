import random

DELTA = 4
DEATH_REWARD = -100

class MOVEMENT:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class NAME:
    SNAKE = "SNAKE"
    FOOD = "FOOD"
    HUMAN = "HUMAN"
    FOODSNAKE = "FOODSNAKE"

class WINDOW:
    WIDTH = 840
    HEIGHT = 840
    ROW = 40
    COL = 40
    BLOCKW = WIDTH // ROW
    BLOCKH = HEIGHT // COL


class GAMESTATE:
    PAUSE = 0
    PLAYING = 1  
    OVER = 2

class COLOR:
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    CYAN = (0, 255, 255)
    MAGENTA = (255, 0, 255)
    colors = [RED, BLUE, CYAN, MAGENTA]

class FOOD:
    COLOR = (0, 255, 0)

class ACTION:
    LEFT_TURN = 0
    RIGHT_TURN = 1
    KEEEP_STRAIGHT = 2


def instantiate(obj=NAME.SNAKE):
    if obj!=NAME.SNAKE:
        return (random.randint(0, WINDOW.ROW), random.randint(0, WINDOW.COL-1))
    else:
        return (random.randint(0+5, WINDOW.ROW-5), random.randint(0+5, WINDOW.COL-5))


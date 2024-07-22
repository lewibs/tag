from objects import Point
from keyboard import make_keyboard_listener
from random import randint

RUNNER = "runner"
CHASER = "CHASER"

CHASER_POSITIONS = {}
RUNNER_POSITIONS = {}

class Agent(Point):
    def __init__(self, engine, type, color, x, y):
        self.engine = engine
        self.is_dead = False
        super().__init__(type, color, x, y)

    def kill(self):
        self.set_color("black")
        self.is_dead = True

    def step(self, x, y):
        if self.id == CHASER:
            CHASER_POSITIONS[f"{self.x},{self.y}"] = False
        elif self.id == RUNNER:
            RUNNER_POSITIONS[f"{self.x},{self.y}"] = False


        if x < 0:
            x = 0
        elif x >= self.engine.width:
            x = self.engine.width - 1

        if y < 0:
            y = 0
        elif y >= self.engine.height:
            y = self.engine.height - 1

        self.set_position(x,y)

        if self.id == CHASER:
            CHASER_POSITIONS[f"{x},{y}"] = True
        elif self.id == RUNNER:
            RUNNER_POSITIONS[f"{x},{y}"] = True

        return x,y

class Runner(Agent):
    def __init__(self, engine, x, y):
        super().__init__(engine, RUNNER, "blue", x, y)

    def step(self):
        if self.is_dead:
            return
        x = self.x + randint(-1, 1)
        y = self.y + randint(-1, 1)
        x,y = super().step(x,y)
        return x, y

class Chaser(Agent):
    def __init__(self, engine, x, y):
        super().__init__(engine, CHASER, "red", x, y)

    def step(self):
        if self.is_dead:
            return
        x = self.x + randint(-1, 1)
        y = self.y + randint(-1, 1)
        x,y = super().step(x,y)
        return x,y
        
class Player(Agent):
    def __init__(self, engine, type, x, y):
        self.keys = make_keyboard_listener("wasd")
        super().__init__(type, engine, "green", x, y)

    def step(self):
        if self.is_dead:
            return

        x = self.x
        y = self.y

        if self.keys["w"]:
            y-=1
        
        if self.keys["s"]:
            y+=1

        if self.keys["d"]:
            x+=1

        if self.keys["a"]:
            x-=1

        x,y = super().step(x,y)

        return x,y
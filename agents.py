from objects import Point
from keyboard import make_keyboard_listener
from random import randint
from dist import euclidian, points_on_circle, get_angle_in_radians
from env import X_WATCHING, KILL_REWARD, VIEW_DIST, MUTATION
from models import RunnerModule, ChaserModule, format_positions
import torch
import torch.nn as nn
import copy

RUNNER = "runner"
CHASER = "CHASER"

CHASER_POSITIONS = {}
RUNNER_POSITIONS = {}

def add_to_agent_map(pos, agent):
    if agent.id == CHASER:
        MAP = CHASER_POSITIONS
    elif agent.id == RUNNER:
        MAP = RUNNER_POSITIONS
    else:
        return

    if pos not in MAP:
        MAP[pos] = []
    MAP[pos].append(agent)

def remove_from_agent_map(pos, agent):
    if agent.id == CHASER:
        MAP = CHASER_POSITIONS
    elif agent.id == RUNNER:
        MAP = RUNNER_POSITIONS
    else:
        return
    
    if pos in MAP:
        if len(MAP[pos]) == 1:
            del MAP[pos]
        else:
            try:
                MAP[pos].remove(agent)
            except:
                print("ran into issue")

# runner_module = RunnerModule(X_WATCHING)
# chaser_module = ChaserModule(X_WATCHING)

class Agent(Point):
    def __init__(self, engine, type, color, x, y):
        self.engine = engine
        self.is_dead = False
        self.memory = []
        self.steps = 0
        self.model = None
        super().__init__(type, color, x, y)

    def save_weights(self, name):
        if self.model:
            torch.save(self.model.state_dict(), name)

    def load_weights(self, weights=None):
        std = MUTATION
        if weights:
            # Make a copy of the provided weights
            copied_weights = copy.deepcopy(weights)
            
            # Load the provided weights into the model
            self.model.load_state_dict(copied_weights)
            
            # Slightly alter the weights
            with torch.no_grad():
                for param in self.model.parameters():
                    param.add_(torch.randn(param.size()) * std)

    def kill(self):
        if f"{self.x},{self.y}" in self.engine.positions_to_object:
            if len(self.engine.positions_to_object[f"{self.x},{self.y}"]) == 1:
                del self.engine.positions_to_object[f"{self.x},{self.y}"]
                remove_from_agent_map(f"{self.x},{self.y}", self)
            else:
                del self.engine.positions_to_object[f"{self.x},{self.y}"][self]
                if all([agent.id != self.id for agent in self.engine.positions_to_object[f"{self.x},{self.y}"]]):
                    remove_from_agent_map(f"{self.x},{self.y}", self)

        if self in self.engine.object_to_position:
            del self.engine.object_to_position[self]

        self.set_color("white")
        self.is_dead = True

    def k_nearest_agents(self, x_nearest):
        GOOD_KEY = 1
        BAD_KEY = -1

        #TODO consider removing all neutral keys to reduce the data that the point needs to consider
        NEUTRAL_KEY = 0

        points = []

        # for key in self.engine.boarder_points:
        #     points.append([BAD_KEY, key[0], key[1]])

        for key in CHASER_POSITIONS.keys():
            data = [int(v) for v in key.split(",")]
            key = BAD_KEY if self.id == RUNNER else NEUTRAL_KEY
            data.insert(0, key)
            if key != NEUTRAL_KEY:
                points.append(data)

        for key in RUNNER_POSITIONS.keys():
            data = [int(v) for v in key.split(",")]
            key = NEUTRAL_KEY if self.id == CHASER else GOOD_KEY
            data.insert(0, key)
            if key != NEUTRAL_KEY:
                points.append(data)

        objects = []
        for pos in points:
            angle = get_angle_in_radians([self.x, self.y], [pos[1], pos[2]])
            dist = euclidian(self.x, self.y, pos[1], pos[2])
            if dist <= VIEW_DIST:
                objects.append([angle, dist*pos[0]])

        objects.sort(key=lambda a:a[1], reverse=True)
        return objects[0:x_nearest]


    def step(self, x, y):
        if self.is_dead:
            raise Exception("This agent is dead")
        
        pos_key = f"{self.x},{self.y}"
        remove_from_agent_map(pos_key, self)

        if x < 0:
            x = 0
        elif x >= self.engine.width:
            x = self.engine.width - 1

        if y < 0:
            y = 0
        elif y >= self.engine.height:
            y = self.engine.height - 1

        self.set_position(x,y)
        pos = f"{x},{y}" 
        add_to_agent_map(pos, self)

        self.steps += 1
        return x,y

class Runner(Agent):
    def __init__(self, engine, x, y, weights=None):
        super().__init__(engine, RUNNER, "blue", x, y)
        runner_model = RunnerModule(X_WATCHING)
        self.model = runner_model
        self.load_weights(weights)

    def step(self):
        if self.is_dead:
            return
        nearest = self.k_nearest_agents(X_WATCHING)
        nearest = format_positions(nearest, X_WATCHING)
        res = self.model(nearest)[0]
        self.memory.append(res)
        x = self.x + int(res[0].item())
        y = self.y + int(res[1].item())
        x,y = super().step(x,y)
        return x, y

class Chaser(Agent):
    def __init__(self, engine, x, y, weights=None):
        super().__init__(engine, CHASER, "red", x, y)
        chaser_model = ChaserModule(X_WATCHING)
        self.model = chaser_model
        self.lifeline = KILL_REWARD
        self.load_weights(weights)

    def step(self):
        if self.is_dead:
            return
        nearest = self.k_nearest_agents(X_WATCHING)
        nearest = format_positions(nearest, X_WATCHING)
        res = self.model(nearest)[0]
        self.memory.append(res)
        x = self.x + int(res[0].item())
        y = self.y + int(res[1].item())
        x,y = super().step(x,y)
        self.lifeline -= 1
        return x,y
        
class Player(Agent):
    def __init__(self, engine, type, x, y):
        self.keys = make_keyboard_listener("wasd")
        self.lifeline = 100
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
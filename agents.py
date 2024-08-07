import pygame
import math
import random
from collections import deque
from enum import Enum
from pynput import keyboard
import torch
from env import MEMORY, K_WATCHING, DEVICE
from models import ChaserModule, RunnerModule

class Agent:
    def __init__(self, color, size, start_pos):
        self.size = size
        self.color = color
        self.position = start_pos
        self.epsilon = 0
        self.n_games = 0
        self.gamma = 0
        self.memory = deque(maxlen=MEMORY)
        self.action = []
        self.model = lambda game_state:torch.tensor([[random.randint(0, 1), random.randint(0, 1)]], dtype=torch.float32, device=DEVICE)

    def draw(self, display):
        # Draw a red rectangle at position (50, 50) with width 100 and height 100
        pygame.draw.circle(display, self.color, self.position, self.size)

    def remember(self, engine, reward):
        self.action.append(reward)
        self.action.append(self.game_state(engine))
        # state, action, reward, next_state, done?
        self.memory.append(self.action)
        self.action = []

    def train(self):
        print(self.memory)

    def game_state(self, engine):
        other_objects = []
        
        for object in engine.objects:
            if object != self:
                other_objects.append(self.dist_to_object(object))

        game_state = [self.position[0], self.position[1], engine.clock.get_time(), engine.h, engine.w]
        
        other_objects.sort(key=lambda a:a[0])

        for object in other_objects[:K_WATCHING]:
            game_state.append(object[0])
            game_state.append(object[1])

        return torch.tensor([game_state], dtype=torch.float32, device=DEVICE)

    def step(self, engine): #TODO reward score
        action = self.model(self.game_state(engine))
        self.action.append(self.game_state(engine))
        self.action.append(action)
        new_x = self.position[0] + action[0][0].item()
        new_y = self.position[1] + action[0][1].item()

        if new_x < self.size:
            new_x = self.size
        
        if new_y < self.size:
            new_y = self.size

        if new_y > engine.h-self.size:
            new_y = engine.h-self.size

        if new_x > engine.w-self.size:
            new_x = engine.w-self.size

        self.position = (new_x, new_y)
        
    def is_touching(self, object):
        dist = math.sqrt(
            math.pow(object.position[0] - self.position[0], 2)+
            math.pow(object.position[1] - self.position[1], 2)
        )

        return dist < self.size + object.size
    
    def dist_to_object(self, object):
        self_x, self_y = self.position
        other_x, other_y = object.position
        
        # Calculate the Euclidean distance
        dist_to = math.sqrt((other_x - self_x) ** 2 + (other_y - self_y) ** 2)
        
        # Calculate the heading in radians
        delta_x = other_x - self_x
        delta_y = other_y - self_y
        heading_rad = math.atan2(delta_y, delta_x)  # atan2 returns angle in radians
        
        # Convert heading to degrees and normalize to [0, 360)
        heading_deg = math.degrees(heading_rad)
        heading_deg = (heading_deg + 360) % 360
        
        return [dist_to, heading_deg]

class User(Agent):
    def __init__(self, color, size, start_pos):
        super().__init__(color, size, start_pos)
        self.current_action = (0, 0)
        self.pressed_keys = set()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.model = lambda game_state:torch.tensor([self.current_action], dtype=torch.float32, device=DEVICE)

    def on_press(self, key):
        try:
            self.pressed_keys.add(key.char)
        except AttributeError:
            pass  # Handle special keys if needed
        self.update_action()

    def on_release(self, key):
        try:
            self.pressed_keys.discard(key.char)
        except AttributeError:
            pass  # Handle special keys if needed
        self.update_action()

    def update_action(self):
        action = [0, 0]
        if 'w' in self.pressed_keys:
            action[1] = -1
        if 's' in self.pressed_keys:
            action[1] = 1
        if 'a' in self.pressed_keys:
            action[0] = -1
        if 'd' in self.pressed_keys:
            action[0] = 1
        
        self.current_action = (action[0], action[1])

class Runner(Agent):
    def __init__(self, color, size, start_pos):
        super().__init__(color, size, start_pos)
        self.model = RunnerModule()
    
class Chaser(Agent):
    def __init__(self, color, size, start_pos):
        super().__init__(color, size, start_pos)
        self.model = RunnerModule()
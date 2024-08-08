import pygame
import math
import random
from collections import deque
from enum import Enum
from pynput import keyboard
import torch
from torch import nn, optim
from env import MEMORY, K_WATCHING, DEVICE, BATCH, TRAINING_GAMES, MIN_EPSILON, START_EPSILON, GAMMA, LR
from models import ChaserModule, RunnerModule

def get_epsilon_linear(n_games):
    epsilon_decay = (START_EPSILON - MIN_EPSILON) / TRAINING_GAMES
    return max(MIN_EPSILON, START_EPSILON - epsilon_decay * n_games)

def random_action(game_state):
    return torch.tensor([random.randint(-1, 1), random.randint(-1, 1)], dtype=torch.float32, device=DEVICE)

def no_action(game_state):
    return torch.tensor([0, 0], dtype=torch.float32, device=DEVICE)

def max_dist(engine):
    return math.sqrt(engine.h**2+engine.w**2)

class Agent:
    def __init__(self, color, size, start_pos):
        self.size = size
        self.color = color
        self.position = start_pos
        self.is_alive = True
        self.memory = deque(maxlen=MEMORY)
        self.action = []
        self.model = random_action
        self.optimizer = None
        self.criterion = None

    def draw(self, display):
        # Draw a red rectangle at position (50, 50) with width 100 and height 100
        pygame.draw.circle(display, self.color, self.position, self.size)

    def remember(self, engine):
        reward = self.reward(engine)
        self.action.append(reward)
        self.action.append(self.game_state(engine))
        # state, action, reward, next_state, done?
        self.memory.append(self.action)
        self.action = []

    def reward(self, engine):
        return torch.tensor(1)

    def train(self):
        if self.optimizer == None or self.criterion == None:
            return

        if len(self.memory) > BATCH:
            sample = random.sample(self.memory, BATCH)
        else:
            sample = self.memory

        state, action, reward, next_state = zip(*sample)

        state = torch.stack(state)
        action = torch.stack(action)
        reward = torch.stack(reward)
        next_state = torch.stack(next_state)

        # 1: predicted Q values with current state
        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(state)):
            Q_new = reward[idx] + GAMMA * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


    def game_state(self, engine):
        other_objects = []
        
        for object in engine.objects:
            if object != self:
                other_objects.append(self.dist_to_object(object))
        
        game_state = [self.position[0] / max_dist(engine), self.position[1] / max_dist(engine)] #engine.clock.get_time(), engine.h, engine.w
        
        other_objects.sort(key=lambda a:a[0])

        for object in other_objects[:K_WATCHING]:
            game_state.append(object[0] / max_dist(engine))
            game_state.append(object[1] / max_dist(engine))

        return torch.tensor(game_state, dtype=torch.float32, device=DEVICE)

    def step(self, engine): #TODO reward score
        game_state = self.game_state(engine)
        epsilon = get_epsilon_linear(engine.n_games)
        random_float = random.random()

        if  random_float < epsilon:
            action = random_action(game_state)
        else:
            action = self.model(self.game_state(engine))

        self.action.append(game_state)
        self.action.append(action)

        new_x = self.position[0] + action[0].item()
        new_y = self.position[1] + action[1].item()

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
        # self_x, self_y = self.position
        # other_x, other_y = object.position
        
        # # Calculate the Euclidean distance
        # dist_to = math.sqrt((other_x - self_x) ** 2 + (other_y - self_y) ** 2)
        
        # # Calculate the heading in radians
        # delta_x = other_x - self_x
        # delta_y = other_y - self_y
        # heading_rad = math.atan2(delta_y, delta_x)  # atan2 returns angle in radians
        
        # # Convert heading to degrees and normalize to [0, 360)
        # heading_deg = math.degrees(heading_rad)
        # heading_deg = (heading_deg + 360) % 360
        
        # return [dist_to, heading_deg]
        return [object.position[0], object.position[1]]

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
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def reward(self, engine):
        if len(self.memory) < 2:
            return torch.tensor(0, dtype=torch.float32, device=DEVICE)
        
        dists = []
        for game_state in list(self.memory)[-2:]:
            game_state = game_state[0]
            dist = math.sqrt((game_state[2]-game_state[0])**2 + (game_state[3]-game_state[1])**2)
            normal_dist = dist
            dists.append(normal_dist)

        dist_diffs = [dists[i] - dists[i+1] for i in range(len(dists) - 1)]
        rewards = [-1*min(0, diff) for diff in dist_diffs]
        reward = sum(rewards)

        return torch.tensor(reward, dtype=torch.float32, device=DEVICE)
    
    def step(self, engine): #TODO reward score
        game_state = self.game_state(engine)
        action = no_action(engine)
        self.action.append(game_state)
        self.action.append(action)
    
class Chaser(Agent):
    def __init__(self, color, size, start_pos):
        super().__init__(color, size, start_pos)
        self.model = ChaserModule()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()

    def reward(self, engine):
        if len(self.memory) < 2:
            return torch.tensor(0, dtype=torch.float32, device=DEVICE)
        dists = []
        for game_state in list(self.memory)[-2:]:
            game_state = game_state[0]
            dist = math.sqrt((game_state[2]-game_state[0])**2 + (game_state[3]-game_state[1])**2)
            normal_dist = dist
            dists.append(normal_dist)

        dist_diffs = [dists[i] - dists[i+1] for i in range(len(dists) - 1)]
        reward = sum(dist_diffs)

        return torch.tensor(reward, dtype=torch.float32, device=DEVICE)
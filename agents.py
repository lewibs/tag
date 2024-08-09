import pygame
import math
import random
from collections import deque
from enum import Enum
from pynput import keyboard
import torch
from torch import nn, optim
from env import MEMORY, K_WATCHING, DEVICE, BATCH, TRAINING_GAMES, MIN_EPSILON, START_EPSILON, GAMMA, LR, LEN_GAME_STATE, DECAY_EPSILON
from models import ChaserModule, RunnerModule
import time

def get_epsilon_linear(n_games):
    epsilon_decay = DECAY_EPSILON
    return max(MIN_EPSILON, START_EPSILON - epsilon_decay * n_games)

def random_action(game_state):
    return torch.tensor([random.randint(-1, 1), random.randint(-1, 1)], dtype=torch.float32, device=DEVICE)

def no_action(game_state):
    return torch.tensor([0, 0], dtype=torch.float32, device=DEVICE)

def max_dist(engine):
    # return 1
    return math.sqrt(engine.h**2+engine.w**2)

def get_direction(radians):
    # Normalize the angle to be within [0, 2π]
    radians = radians % (2 * math.pi)
    # Calculate X and Y offsets using cos and sin
    x_offset = round(math.cos(radians))
    y_offset = round(math.sin(radians))
   
    return [x_offset, y_offset]

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

    def reset(self, start_pos):
        self.position = start_pos

    def draw(self, display):
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
        if self.optimizer is None or self.criterion is None:
            return

        if len(self.memory) > BATCH:
            sample = random.sample(self.memory, BATCH)
        else:
            sample = self.memory

        state, action, reward, next_state = zip(*sample)

        state = torch.stack(state)
        action = torch.stack(action)
        reward = torch.tensor(reward).unsqueeze(1)  # reward should have the same batch size
        next_state = torch.stack(next_state)

        # 1: predicted Q values with current state
        pred = self.model(state)

        # 2: Q_new = r + γ * max(next_predicted Q value) -> only do this if not done
        with torch.no_grad():  # No gradient should flow through the target computation
            next_pred = self.model(next_state)
            target = pred.clone()
            for idx in range(len(state)):
                Q_new = reward[idx] + GAMMA * torch.max(next_pred[idx])
                target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3: Backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(pred, target.detach())  # detach the target from the graph
        loss.backward()
        self.optimizer.step()

        # print(self.memory[-1])


    def game_state(self, engine):
        game = torch.tensor([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ], dtype=torch.float32, device=DEVICE)
        
        self_position = [self.position[0] / engine.w, self.position[1] / engine.h]
        for object in engine.objects:
            if object != self:
                object_position = [object.position[0] / engine.w, object.position[1] / engine.h]
                delta_position = [self_position[0] - object_position[0], self_position[1] - object_position[1]] #easier then using radians

                if delta_position[0] < -(self.size*1.5/engine.w):
                    x = 2
                elif delta_position[0] > (self.size*1.5/engine.w):
                    x = 0
                else:
                    x = 1

                if delta_position[1] < -(self.size*1.5/engine.h):
                    y = 2
                elif delta_position[1] > (self.size*1.5/engine.h):
                    y = 0
                else:
                    y = 1

                game[y][x] = 1

        if isinstance(self, Runner):
            print(game)

        return game.flatten()

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
        self_x, self_y = self.position
        other_x, other_y = object.position
        
        # Calculate the Euclidean distance
        dist_to = math.sqrt((other_x - self_x) ** 2 + (other_y - self_y) ** 2)
        
        # Calculate the heading in radians
        delta_x = other_x - self_x
        delta_y = other_y - self_y
        heading_rad = math.atan2(delta_y, delta_x)  # atan2 returns angle in radians

        return dist_to, heading_rad

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
        #get the x,y position of the target location
        game_state = self.action[0].clone()
        index = torch.nonzero(game_state, as_tuple=True)[0].item()
        target_x = int(index // math.sqrt(LEN_GAME_STATE))
        target_y = int(index % math.sqrt(LEN_GAME_STATE))

        #get the models x,y position that it chose
        game_action = self.action[1].clone()
        actual_x = int(game_action[0].item())
        actual_y = int(game_action[1].item())

        if actual_x == 0 and actual_y == 0:
            reward = 1000
        else:
            reward = -1

        return torch.tensor(reward, dtype=torch.float32, device=DEVICE)

        if game_action[0].item() < -0.25:
            delta_x = -1
        elif game_action[0].item() > 0.25:
            delta_x = 1
        else:
            delta_x = 0
        
        action_x = 1 + delta_x

        if int(game_action[1].item()) == -1:
            delta_y = -1
        elif int(game_action[1].item()) == 1:
            delta_y = 1
        else:
            delta_y = 0

        action_y = 1 + delta_y

        print("reward")
        print(game_state)
        print(game_action)
        print(target_x, target_y)
        print(action_x, action_y)

        if action_y == 1 and action_x == 1 and target_y == 1 and target_x == 1:
            reward = 1
        elif action_y == 1 and action_x == 1:
            reward = -1
        else:
            dist = math.sqrt((target_x-action_x)**2 + (target_y-action_y)**2)
            ave_dist = math.sqrt((math.sqrt(LEN_GAME_STATE)//2)**2 + (math.sqrt(LEN_GAME_STATE)//2)**2)
            reward = ave_dist - dist

        print(reward)

        return torch.tensor(reward, dtype=torch.float32, device=DEVICE)

#TODO I think there is a problem, that since the cords are swapped, it will almost always either run away or not find the desired outcome?
import pygame
from env import TIME_LIMIT, GAME_SPEED
import random

class Engine:
    def __init__(self, w=500, h=500, time_limit=TIME_LIMIT, game_speed=GAME_SPEED):
        pygame.init()  # Initialize pygame
        self.w = w
        self.h = h
        self.game_speed = game_speed
        self.time_limit = time_limit
        self.display = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.objects = []
        self.n_games = 1
        self.reset()

    def reset(self):
        for object in self.objects:
            object.reset((random.randint(0, self.w), random.randint(0, self.h)))
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()  # Reset start time
        self.n_games += 1

    def add(self, object):
        self.objects.append(object)

    def step(self):
        for object in self.objects:
            object.step(self)
            
        self.clock.tick(self.game_speed)
    
    def update_memories(self):
        for object in self.objects:
            object.remember(self)
    
    def train(self):
        for object in self.objects:
            object.train()

    def render(self):
        # Clear the screen with a black color
        self.display.fill((0, 0, 0))

        for object in self.objects:
            object.draw(self.display)    

        # Update the display
        pygame.display.flip()
        
    def apply_game_logic(self):
        pass

    def start(self):
        while (pygame.time.get_ticks() - self.start_time) < self.time_limit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
            
            self.step()
            self.apply_game_logic()
            self.render()
            self.update_memories()
            self.train()
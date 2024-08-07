import pygame
from enum import Enum

class Engine:
    def __init__(self, w=500, h=500, time_limit=6, game_speed=40):
        pygame.init()  # Initialize pygame
        self.w = w
        self.h = h
        self.game_speed = game_speed
        self.time_limit = time_limit * 1000  # Convert to milliseconds
        self.display = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.objects = []
        self.reset()

    def reset(self):
        self.objects = []
        self.clock = pygame.time.Clock()
        self.start_time = pygame.time.get_ticks()  # Reset start time

    def add(self, object):
        self.objects.append(object)

    def step(self):
        for object in self.objects:
            object.step(self)
            
        self.clock.tick(self.game_speed)
    
    def update_memories(self):
        for object in self.objects:
            object.remember(self, 1)
    
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
from graphics import *
import time

class Engine:
    def __init__(self, height, width):
        self.running = True
        self.height = height
        self.width = width
        self.objects = []
        self.callbacks = []
        self.win = GraphWin(width = self.height, height = self.width)

    def pause(self):
        #make pause freeze the game
        self.running = False

    def render_loop(self):
        while self.running:
            for callback in self.callbacks:
                callback(self)
            self.draw()
            time.sleep(0.1)
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
        
    def add_object(self, object):
        self.objects.append(object)

    def draw(self):
        for object in self.objects:
            object.draw(self.win)

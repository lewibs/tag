from graphics import *
import time
from env import GAME_SPEED

class Engine:
    def __init__(self, height, width):
        self.boarder_points = []
        self.running = True
        self.height = height
        self.width = width
        self.time = 0
        self.objects = []
        self.positions_to_object = {}
        self.object_to_position = {}
        self.callbacks = []
        self.win = GraphWin(width = self.height, height = self.width)

        for i in range(self.height):
            #left boarder
            self.boarder_points.append([0,i])
            #right boarder
            self.boarder_points.append([self.width,i])

        for i in range(self.height):
            #bottom boarder
            self.boarder_points.append([i,0])
            #top boarder
            self.boarder_points.append([i,self.height])

    def pause(self):
        #make pause freeze the game
        self.running = False

    def render_loop(self):
        while self.running:
            for callback in self.callbacks:
                callback(self)
            self.time += 1
            self.draw()
            time.sleep(GAME_SPEED)
    
    def add_callback(self, callback):
        self.callbacks.append(callback)
        
    def add_object(self, object):
        self.objects.append(object)

    def remove_object(self, object):
        if object in self.objects:
            self.objects.remove(object)
        if object in self.object_to_position:
            pos_old = self.object_to_position[object]
            # Ensure the old position and object exist in the dictionary before deleting
            if pos_old in self.positions_to_object:
                if object in self.positions_to_object[pos_old]:
                    del self.positions_to_object[pos_old][object]
                    if len(self.positions_to_object[pos_old]) == 0:
                        del self.positions_to_object[pos_old]
        object.undraw()
        

    def draw(self):
        for object in self.objects:
            if object.needs_update:
                pos = f"{object.x},{object.y}"


                if object in self.object_to_position:
                    pos_old = self.object_to_position[object]
                    # Ensure the old position and object exist in the dictionary before deleting
                    if pos_old in self.positions_to_object:
                        if object in self.positions_to_object[pos_old]:
                            del self.positions_to_object[pos_old][object]
                            if len(self.positions_to_object[pos_old]) == 0:
                                del self.positions_to_object[pos_old]

                # Ensure the current position key exists in the dictionary
                if pos not in self.positions_to_object:
                    self.positions_to_object[pos] = {}

                self.object_to_position[object] = pos
                self.positions_to_object[pos][object] = True

            object.draw(self.win)


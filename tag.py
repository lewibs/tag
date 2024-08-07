from engine import Engine
from agents import Runner, Chaser
import random

AGENT_SIZE = 50
WIDTH = 500
HEIGHT = 500
RED = (255,0,0)
BLUE = (0,0,255)

engine = Engine(w=WIDTH, h=HEIGHT)

rand_pos1 = (random.randint(0, WIDTH), random.randint(0, HEIGHT)) 
rand_pos2 = (random.randint(0, WIDTH), random.randint(0, HEIGHT)) 
runner = Runner(color=BLUE, size=AGENT_SIZE, start_pos=rand_pos1)
chaser = Chaser(color=RED, size=AGENT_SIZE, start_pos=rand_pos2)
engine.add(runner)
engine.add(chaser)
engine.start()
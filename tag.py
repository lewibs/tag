from engine import Engine
from agents import Runner, Chaser
import random
from env import AGENT_SIZE, WIDTH, HEIGHT, RED, BLUE, TRAINING_GAMES

engine = Engine(w=WIDTH, h=HEIGHT)

while engine.n_games < TRAINING_GAMES:
    rand_pos1 = (random.randint(0, WIDTH), random.randint(0, HEIGHT)) 
    rand_pos2 = (random.randint(0, WIDTH), random.randint(0, HEIGHT)) 
    runner = Runner(color=BLUE, size=AGENT_SIZE, start_pos=rand_pos1)
    chaser = Chaser(color=RED, size=AGENT_SIZE, start_pos=rand_pos2)
    engine.add(runner)
    engine.add(chaser)
    engine.start()
    engine.reset()
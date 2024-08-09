from engine import Engine
from agents import Runner, Chaser
import random
from env import AGENT_SIZE, WIDTH, HEIGHT, RED, BLUE, TRAINING_GAMES
from agents import get_epsilon_linear

rand_pos1 = (random.randint(0, WIDTH), random.randint(0, HEIGHT)) 
rand_pos2 = (random.randint(0, WIDTH), random.randint(0, HEIGHT)) 

engine = Engine(w=WIDTH, h=HEIGHT)
runner = Runner(color=BLUE, size=AGENT_SIZE, start_pos=rand_pos1)
chaser = Chaser(color=RED, size=AGENT_SIZE, start_pos=rand_pos2)
engine.add(runner)
engine.add(chaser)

while engine.n_games < TRAINING_GAMES:
    print(engine.n_games, get_epsilon_linear(engine.n_games))
    engine.start()
    engine.reset()
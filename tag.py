from engine import Engine
from agents import Runner, Chaser
import random
from env import AGENT_SIZE, WIDTH, HEIGHT, RED, BLUE, TRAINING_GAMES
from agents import get_epsilon_linear, chaser_reward
import matplotlib.pyplot as plt
import numpy as np

def graph(chaser_reward):
    plt.clf()  # Clear the current figure
    x_values = np.arange(len(chaser_reward))
    plt.plot(x_values, chaser_reward, label='Chaser Reward', marker='o')
    z = np.polyfit(x_values, chaser_reward, 1)
    p = np.poly1d(z)
    plt.plot(x_values, p(x_values), "r--", label='Trendline')
    plt.xlabel('Index')
    plt.ylabel('Reward Value')
    plt.title('Chaser Reward with Trendline')
    plt.legend()
    plt.pause(0.1)  # Pause briefly to update the plot

# Example loop
plt.ion()  # Turn on interactive mode

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
    graph(chaser_reward)
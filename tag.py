from engine import Engine
from agents import Runner, Chaser
import random
from env import AGENT_SIZE, WIDTH, HEIGHT, RED, BLUE, TRAINING_GAMES
from agents import get_epsilon_linear, chaser_reward
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description="Run training for the chaser and runner models.")
parser.add_argument('--mode', type=str, default='train', help='Mode to run: "train" or "skip"')
args = parser.parse_args()

def graph(chaser_reward):
    if not chaser_reward:  # Check if chaser_reward is empty
        print("chaser_reward is empty, skipping graphing.")
        return

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

runner_model_path = "runner_model.pth"
chaser_model_path = "chaser_model.pth"

if args.mode == "train":
    while engine.n_games < TRAINING_GAMES:
        print(engine.n_games, get_epsilon_linear(engine.n_games))
        engine.start()
        engine.reset()
        graph(chaser_reward)

    torch.save(runner.model.state_dict(), runner_model_path)
    torch.save(chaser.model.state_dict(), chaser_model_path)

engine = Engine(w=WIDTH, h=HEIGHT, time_limit=100000)
runner = Runner(color=BLUE, size=AGENT_SIZE, start_pos=rand_pos1)
chaser = Chaser(color=RED, size=AGENT_SIZE, start_pos=rand_pos2)
engine.add(runner)
engine.add(chaser)

# Load the saved model state dictionary
model_state = torch.load(runner_model_path)
runner.model.load_state_dict(model_state)
model_state = torch.load(chaser_model_path)
runner.model.load_state_dict(model_state)
# Set the model to evaluation mode if you're using it for inference
runner.model.eval()
chaser.model.eval()

engine.start()
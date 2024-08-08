from engine import Engine
from objects import Point
from agents import Player, Chaser, Runner, RUNNER, CHASER, CHASER_POSITIONS, KILL_REWARD
from random import randint
from env import GAME_SIZE, PLAYERS, TAKE_X_WINNERS, TRAINING_EPOCHS, REGEN_RATE
import sys
import random
import torch

def load_weights(paths=[]):
    weights = []

    for path in paths:
        weights.append(torch.load(path))

    return weights

def run_game(train=0, player=None, runner_genetics=None, chaser_genetics=None):
    engine = Engine(GAME_SIZE,GAME_SIZE)
    agents = []
    runners = []
    chasers = []

    if player:
        player = Player(CHASER, engine, engine.height/2, engine.width/2)
        agents.append(player)
        chasers.append(player)

    runner_weights = load_weights(runner_genetics) if runner_genetics else [None]
    chaser_weights = load_weights(chaser_genetics) if chaser_genetics else [None]

    def make_chaser(chaser_weights):
        chaser_weight = random.choice(chaser_weights)
        chaser = Chaser(engine, randint(0, engine.height), randint(0, engine.width), chaser_weight)
        chasers.append(chaser)
        agents.append(chaser)
        return chaser

    def make_runner(runner_weights):
        runner_weight = random.choice(runner_weights)
        runner = Runner(engine, randint(0, engine.height), randint(0, engine.width), runner_weight)
        runners.append(runner)
        agents.append(runner)
        return runner

    def remove_agent(agent):
        if agent.id == RUNNER:
            runners.remove(agent)
        elif agent.id == CHASER:
            chasers.remove(agent)
        agents.remove(agent)

    for i in range(PLAYERS):
        make_chaser(chaser_weights)
        make_runner(runner_weights)
        
    for agent in agents: 
        engine.add_object(agent)

    def handle_agents(engine):
        is_runner_alive = False
        is_chaser_alive = False
        nonlocal train

        for agent in agents:
            if agent.is_dead:
                continue

            #check to kill any
            # if agent.x <= 0 or agent.y <= 0 or agent.x >= engine.width - 1 or agent.y >= engine.height -1:
            #     agent.kill()
            #     remove_agent(agent)
            #     continue 

            #check to kill chaser
            if agent.id == CHASER:
                is_chaser_alive = True
                if agent.lifeline <= 0:
                    agent.kill()
                    remove_agent(agent)
                    continue

            #check to kill runner
            if agent.id == RUNNER:
                is_runner_alive = True

                #we are checking if the runner has anything around it. if it has been tagged
                pos = [
                    f"{agent.x},{agent.y}",

                    f"{agent.x},{agent.y-1}",
                    f"{agent.x+1}{agent.y-1}",
                    f"{agent.x+1}{agent.y+1}",
                    f"{agent.x+1},{agent.y}",
                    f"{agent.x-1},{agent.y}",
                    f"{agent.x-1},{agent.y+1}",
                    f"{agent.x-1},{agent.y+1}",
                    f"{agent.x},{agent.y+1}",

                    f"{agent.x},{agent.y-2}",
                    f"{agent.x+2}{agent.y-2}",
                    f"{agent.x+2}{agent.y+2}",
                    f"{agent.x+2},{agent.y}",
                    f"{agent.x-2},{agent.y}",
                    f"{agent.x-2},{agent.y+2}",
                    f"{agent.x-2},{agent.y+2}",
                    f"{agent.x},{agent.y+2}",

                    # f"{agent.x},{agent.y-3}",
                    # f"{agent.x+3}{agent.y-3}",
                    # f"{agent.x+3}{agent.y+3}",
                    # f"{agent.x+3},{agent.y}",
                    # f"{agent.x-3},{agent.y}",
                    # f"{agent.x-3},{agent.y+3}",
                    # f"{agent.x-3},{agent.y+3}",
                    # f"{agent.x},{agent.y+3}",
                ]
                
                for pos in pos:
                    if pos in CHASER_POSITIONS and CHASER_POSITIONS[pos]:
                        if pos in CHASER_POSITIONS:
                            temp = [agent for agent in CHASER_POSITIONS[pos] if agent.id == CHASER]
                            for chaser in temp:
                                chaser.lifeline += KILL_REWARD
                        agent.kill()
                        remove_agent(agent)
                        break
            agent.step()

        if train != 0 and engine.time % REGEN_RATE == 0:
            train -= 1
            top_runners = []
            top_chasers = []
            if len(runners) >= TAKE_X_WINNERS:
                runners.sort(key=lambda a:a.steps, reverse=True)
                top_runners = runners[0:TAKE_X_WINNERS]
            if len(chasers) >= TAKE_X_WINNERS:
                chasers.sort(key=lambda a:a.lifeline+a.steps*2, reverse=True)
                top_chasers = chasers[0:TAKE_X_WINNERS]

            chaser_weights = []
            runner_weights = []
            
            for i in range(TAKE_X_WINNERS):
                if len(top_chasers) >= TAKE_X_WINNERS:
                    chaser = top_chasers[i]
                    chaser_weights.append(chaser.model.state_dict())

                if len(top_runners) >= TAKE_X_WINNERS:
                    runner = top_runners[i]
                    runner_weights.append(runner.model.state_dict())

            if len(chaser_weights) != 0:
                chaser = make_chaser(chaser_weights)
                engine.add_object(chaser)

            if len(runner_weights) != 0:
                runner = make_runner(runner_weights)
                engine.add_object(runner)
            
            
        if not is_runner_alive or not is_chaser_alive:
            engine.pause()

    engine.add_callback(handle_agents)
    engine.render_loop()

    runners.sort(key=lambda a:a.steps, reverse=True)
    chasers.sort(key=lambda a:a.lifeline+a.steps*2, reverse=True)

    return [chasers[0:TAKE_X_WINNERS], runners[0:TAKE_X_WINNERS]]

if __name__ == "__main__":
    args = {}

    for arg in sys.argv:
        args[arg] = True

    player = True if "player" in args else False
    training = True if "train" in args else False

    loops = TRAINING_EPOCHS
    if training:
        top_chasers, top_runners = run_game(train=loops, player=player)
        for i in range(TAKE_X_WINNERS):
            if i<len(top_chasers):
                chaser = top_chasers[i]
                chaser_path = f"./weights/chaser_{i}.pth"
                chaser.save_weights(chaser_path)
            if i<len(top_runners):
                runner = top_runners[i]
                runner_path = f"./weightsrunner_{i}.pth"
                runner.save_weights(runner_path)

    else:
        chaser_weights = load_weights(["./weights/chaser_0.pth"])
        runner_weights = load_weights(["./weights/runner_0.pth"])
        run_game(player=player, chaser_genetics=chaser_weights, runner_genetics=runner_weights)
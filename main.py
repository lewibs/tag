from engine import Engine
from objects import Point
from agents import Player, Chaser, Runner, RUNNER, CHASER, CHASER_POSITIONS, KILL_REWARD
from random import randint
from env import GAME_SIZE, PLAYERS, TAKE_X_WINNERS, TRAINING_EPOCHS, VIEW_DIST
import sys
import random
import torch


def run_game(player=None, runner_genetics=None, chaser_genetics=None):
    engine = Engine(GAME_SIZE,GAME_SIZE)
    agents = []
    runners = []
    chasers = []

    if player:
        player = Player(CHASER, engine, engine.height/2, engine.width/2)
        agents.append(player)
        chasers.append(player)

    runner_weights = []
    chaser_weights = []

    if runner_genetics:
        for path in runner_genetics:
            runner_weights.append(torch.load(path))
    else:
        runner_weights = [None]

    if chaser_genetics:
        for path in chaser_genetics:
            chaser_weights.append(torch.load(path))
    else:
        chaser_weights = [None]

    for i in range(PLAYERS):
        runner_weight = random.choice(runner_weights)
        chaser_weight = random.choice(chaser_weights)

        chaser = Chaser(engine, randint(0, engine.height), randint(0, engine.width), chaser_weight)
        chasers.append(chaser)
        runner = Runner(engine, randint(0, engine.height), randint(0, engine.width), runner_weight)
        runners.append(runner)
        agents.append(chaser)
        agents.append(runner)

    for agent in agents: 
        engine.add_object(agent)

    def handle_agents(engine):
        is_runner_alive = False
        is_chaser_alive = False

        for agent in agents:
            if agent.is_dead:
                continue

            #check to kill any
            if agent.x <= 0 or agent.y <= 0 or agent.x >= engine.width - 1 or agent.y >= engine.height -1:
                agent.kill()
                continue 

            #check to kill chaser
            if agent.id == CHASER:
                is_chaser_alive = True
                if agent.lifeline <= 0:
                    agent.kill()
                    continue

            #check to kill runner
            if agent.id == RUNNER:
                is_runner_alive = True
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
                ]
                
                for pos in pos:
                    if pos in CHASER_POSITIONS and CHASER_POSITIONS[pos]:
                        if pos in engine.positions_to_object:
                            temp = [agent for agent in engine.positions_to_object[pos] if agent.id == CHASER]
                            for chaser in temp:
                                chaser.lifeline += KILL_REWARD
                            agent.kill()
                            break
            agent.step()
            
        if not is_runner_alive or not is_chaser_alive:
            engine.pause()

    engine.add_callback(handle_agents)
    engine.render_loop()

    chasers.sort(key=lambda a:a.steps, reverse=True)
    runners.sort(key=lambda a:a.steps, reverse=True)

    return [chasers[0:TAKE_X_WINNERS], runners[0:TAKE_X_WINNERS]]

if __name__ == "__main__":
    args = {}

    for arg in sys.argv:
        args[arg] = True

    player = True if "player" in args else False
    training = True if "train" in args else False

    loops = TRAINING_EPOCHS
    if training:
        c_paths = None
        r_paths = None
        while loops > 0:
            top_chasers, top_runners = run_game(runner_genetics=r_paths, chaser_genetics=c_paths)
            c_paths = []
            r_paths = []
            
            for i in range(TAKE_X_WINNERS):
                chaser = top_chasers[i]
                runner = top_runners[i]
                chaser_path = f"chaser_{i}.pth"
                runner_path = f"runner_{i}.pth"
                chaser.save_weights(f"chaser_{i}.pth")
                runner.save_weights(f"runner_{i}.pth")
                c_paths.append(chaser_path)
                r_paths.append(runner_path)

                top_runner = r_paths
                top_chasers

            loops-=1
    else:
        run_game(player=player)
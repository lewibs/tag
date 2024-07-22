from engine import Engine
from objects import Point
from agents import Player, Chaser, Runner, RUNNER, CHASER, RUNNER_POSITIONS, CHASER_POSITIONS
from keyboard import make_keyboard_listener
from random import randint

engine = Engine(200,200)

agents = [
    Player(CHASER, engine, engine.height/2, engine.width/2)
]
runners = []
chasers = []

for i in range(5):
    chaser = Chaser(engine, randint(0, engine.height), randint(0, engine.width))
    chasers.append(chasers)
    runner = Runner(engine, randint(0, engine.height), randint(0, engine.width))
    runners.append(runner)
    agents.append(chaser)
    agents.append(runner)

for agent in agents: 
    engine.add_object(agent)

def step_agents(engine):
    for agent in agents: 
        agent.step()

def tag_agents(engine):
    for agent in agents:
        if isinstance(agent, Runner):
            pos = f"{agent.x},{agent.y}"
            if pos in CHASER_POSITIONS and CHASER_POSITIONS[pos]:
                agent.kill()
            
def check_for_end_of_game(engine):
    for agent in runners:
        if not agent.is_dead:
            return
    engine.pause()

engine.add_callback(step_agents)
engine.add_callback(tag_agents)
engine.add_callback(check_for_end_of_game)
engine.render_loop()
from engine import Engine
from objects import Point
from agents import Player, Chaser, Runner, RUNNER, CHASER, RUNNER_POSITIONS, CHASER_POSITIONS
from keyboard import make_keyboard_listener
from random import randint
from dist import points_on_circle
from env import VIEW_DIST

engine = Engine(500,500)

player = Player(CHASER, engine, engine.height/2, engine.width/2)
agents = [player]
runners = []
chasers = []

for i in range(50):
    chaser = Chaser(engine, randint(0, engine.height), randint(0, engine.width))
    chasers.append(chasers)
    runner = Runner(engine, randint(0, engine.height), randint(0, engine.width))
    runners.append(runner)
    agents.append(chaser)
    agents.append(runner)

for agent in agents: 
    engine.add_object(agent)

# points = []
# def show_player_sight_grid(engine):
#     global points
#     for point in points:
#         engine.remove_object(point)
    
#     cords = points_on_circle(player.x, player.y, VIEW_DIST, VIEW_DIST*2)
    
#     points = []
#     for point in cords:
#         point = Point("boundry", "pink", point[0], point[1])
#         points.append(point)
#         engine.add_object(point)


def handle_agents(engine):
    is_alive = False
    
    for agent in agents:
        if isinstance(agent, Runner):
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
                    agent.kill()
                    break
    
            if not agent.is_dead:
                is_alive = True
    
        if not agent.is_dead:
            agent.step()
        
    if not is_alive:
        engine.pause()

engine.add_callback(handle_agents)
engine.render_loop()
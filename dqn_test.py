import torch
from dqn import Agent, DQN
import config
import os

if config.on_TACC:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

def runDemo():
    dqn = DQN(4)
    dqn.load_state_dict(torch.load("./model20000.pkl"))
    agent = Agent(dqn)
    agent.test()


'''run'''
if __name__ == '__main__':
    runDemo()

import torch
from DQNNet import DQNNet
from DoubleDQNAgent import DoubleDQNAgent
from DQNAgent import DQNAgent
import config
import os

if config.on_TACC:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

def runDemo():
    dqn = DQNNet()
    dqn.load_state_dict(torch.load("./model190000.pkl"))
    if config.use_double_dqn:
        agent = DoubleDQNAgent(dqn)
    else:
        agent = DQNAgent(dqn)
    agent.test()


'''run'''
if __name__ == '__main__':
    runDemo()

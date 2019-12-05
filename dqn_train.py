from DQNNet import DQNNet
from DQNAgent import DQNAgent
from DoubleDQNAgent import DoubleDQNAgent
# import torch
# torch.manual_seed(9001)
import os
import config

if config.on_TACC:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

def train():
    dqn = DQNNet()
    if config.use_double_dqn:
        agent = DoubleDQNAgent(dqn)
    else:
        agent = DQNAgent(dqn)
    agent.train()


if __name__ == '__main__':
    train()

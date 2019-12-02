from dqn import Agent, DQN
import torch
torch.manual_seed(9001)
import os
import config

if config.on_TACC:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

def train():
    dqn = DQN(4)
    dqn.load_state_dict(torch.load("./1.pkl"))
    agent = Agent(dqn)
    agent.train()


if __name__ == '__main__':
    train()

from DQNNet import DQNNet
from DQNAgent import DQNAgent
from DoubleDQNAgent import DoubleDQNAgent
import torch
# torch.manual_seed(9001)
import os
import config

if config.on_TACC:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

def train():
    dqn = DQNNet()
    dqn.load_state_dict(torch.load("./model120000.pkl", map_location=lambda storage, loc: storage))

    foldname = "model_dqn"
    if config.use_double_dqn:
        foldname += "_doubleQ"
    if config.use_per:
        foldname += "_per"

    if config.use_double_dqn:
        agent = DoubleDQNAgent(dqn)
    else:
        agent = DQNAgent(dqn)
    agent.train(foldname)


if __name__ == '__main__':
    train()

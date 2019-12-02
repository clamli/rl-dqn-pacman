from dqn import Agent, DQN
import os
import config

if config.on_TACC:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

def train():
    dqn = DQN(4)
    agent = Agent(dqn)
    agent.train2()


if __name__ == '__main__':
    train()

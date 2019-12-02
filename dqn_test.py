import torch
from dqn import Agent, DQN


def runDemo():
    dqn = DQN(4)
    dqn.load_state_dict(torch.load("./10000.pkl"))
    agent = Agent(dqn)
    agent.test()


'''run'''
if __name__ == '__main__':
    runDemo()

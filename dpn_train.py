from dqn import Agent, DQN


def train():
    dqn = DQN(4)
    agent = Agent(dqn)
    agent.train()


if __name__ == '__main__':
    train()

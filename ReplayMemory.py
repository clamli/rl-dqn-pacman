import random
import numpy as np
import config

from SumTree import SumTree


class ReplayMemory:

    def __init__(self):
        self.memory = []

    def add(self, state, action, reward, is_gameover, next_state):
        if len(self.memory) >= config.max_memory_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, is_gameover, next_state))

    def sample(self, batch_size):
        rand_ind = random.sample(range(0, len(self.memory)), min(batch_size, len(self.memory)))
        state_lst = [self.memory[i][0] for i in rand_ind]
        action_lst = [self.memory[i][1] for i in rand_ind]
        reward_lst = [self.memory[i][2] for i in rand_ind]
        is_gameover_lst = [self.memory[i][3] for i in rand_ind]
        next_state_lst = [self.memory[i][4] for i in rand_ind]
        return [], state_lst, action_lst, reward_lst, is_gameover_lst, next_state_lst


class PrioritizedReplayMemory:
    e = 0.01
    a = 0.6
    epsilon = 0.01
    abs_err_upper = 1.

    def __init__(self, memory_size):
        self.memory = SumTree(memory_size)

    def add(self, state, action, reward, is_gameover, next_state):
        p = np.max(self.memory.tree[-self.memory.memory_size])
        if p == 0:
            p = self.abs_err_upper
        transition = (state, action, reward, is_gameover, next_state)
        self.memory.add(p, transition)

    def update(self, position, error):
        error += self.epsilon
        error = min(error, self.abs_err_upper)
        p = error ** self.a
        self.memory.update(position, p)

    def sample(self, sample_size):
        state_lst = []
        action_lst = []
        reward_lst = []
        is_gameover_lst = []
        next_state_lst = []
        positions_lst = []
        segment = self.memory.total() / sample_size
        for i in range(sample_size):
            a = segment * i
            b = segment * (i+1)
            s = random.uniform(a, b)
            position, _, data = self.memory.get(s)
            positions_lst.append(position)
            state_lst.append(data[0])
            action_lst.append(data[1])
            reward_lst.append(data[2])
            is_gameover_lst.append(data[3])
            next_state_lst.append(data[4])
        return positions_lst, state_lst, action_lst, reward_lst, is_gameover_lst, next_state_lst
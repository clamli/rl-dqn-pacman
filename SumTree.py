"""
    Modified based on the code: https://github.com/jaara/AI-blog/blob/master/SumTree.py
"""

import numpy as np
import random as rd


class SumTree:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.tree = np.zeros(2*memory_size - 1)
        self.transitions = np.zeros(memory_size, dtype=object)
        self.cur_position = 0
        self.size = 0

    def __len__(self):
        return self.size if self.size <= self.memory_size else self.memory_size

    def add(self, p, transition):
        cur_tree_position = self.cur_position + self.memory_size - 1
        self.transitions[self.cur_position] = transition
        self.update(cur_tree_position, p)
        self.cur_position += 1
        self.size += 1
        if self.cur_position >= self.memory_size:
            self.cur_position = 0

    def update(self, tree_position, p):
        change = p - self.tree[tree_position]
        self.tree[tree_position] = p
        while tree_position != 0:
            tree_position = (tree_position - 1) // 2
            self.tree[tree_position] += change

    def get(self, v):
        cur_tree_position = 0
        while True:
            left_tree_position = 2 * cur_tree_position + 1
            right_tree_position = left_tree_position + 1
            if left_tree_position >= len(self.tree):
                break
            if v <= self.tree[left_tree_position]:
                cur_tree_position = left_tree_position
            else:
                v -= self.tree[left_tree_position]
                cur_tree_position = right_tree_position
        cur_position = cur_tree_position - self.memory_size + 1
        if cur_position >= self.cur_position:
            cur_position = rd.randint(0, max(0, self.cur_position-1))
            cur_tree_position = cur_position + self.memory_size - 1
        return cur_tree_position, self.tree[cur_tree_position], self.transitions[cur_position]

    def total(self):
        return self.tree[0]
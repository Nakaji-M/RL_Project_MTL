import numpy as np
import random
import torch

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        samples = random.sample(self.memory, batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in samples if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in samples if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e[2] for e in samples if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in samples if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in samples if e is not None]).astype(np.uint8)).float()

        return states, actions, rewards, next_states, dones


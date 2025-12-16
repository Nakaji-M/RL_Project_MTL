import torch
import random
from utils.ANN import ANN
import torch.optim as optim
from utils.ReplayMemory import ReplayMemory
import numpy as np
import torch.nn.functional as F

minibatch = 128
gamma = 0.95

class Agent():
    def __init__(self, state_size, action_size):
        self.capacity = 100000
        self.state_size = state_size
        self.action_size = action_size
        self.local_qnetwork = ANN(state_size, action_size)
        self.target_qnetwork = ANN(state_size, action_size)
        self.target_qnetwork.load_state_dict(self.local_qnetwork.state_dict())
        self.optimizer = optim.Adam(self.local_qnetwork.parameters(), lr=1e-3)
        self.memory = ReplayMemory(capacity=self.capacity)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        self.t_step = (self.t_step + 1) % 4

        if len(self.memory.memory) > max(self.capacity / 10, minibatch) and self.t_step == 0:
            experiences = self.memory.sample(minibatch)
            self.learn(experiences, gamma)

    def get_action(self, state, action_size, epsilon=0.5):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.local_qnetwork.eval()
        with torch.no_grad():
            action_values = self.local_qnetwork(state)
        self.local_qnetwork.train()
        if random.random() < epsilon:
            return np.random.choice(range(action_size)) # TODO: Check range action_values
        else:
            return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        next_q_targets = self.target_qnetwork(next_states).detach().max(dim=1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * next_q_targets * (1 - dones))
        q_expected = self.local_qnetwork(states).gather(1, actions)
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.soft_update(self.local_qnetwork, self.target_qnetwork, tau=0.01)

    def soft_update(self, local_qnetwork, target_qnetwork, tau):
        for target_params, local_params in zip(target_qnetwork.parameters(), local_qnetwork.parameters()):
            target_params.data.copy_(tau * local_params.data + (1.0 - tau) * target_params.data)

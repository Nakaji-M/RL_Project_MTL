import torch
import random
from utils.ANN import ANN
import torch.optim as optim
from utils.ReplayMemory import ReplayMemory
import numpy as np
import torch.nn.functional as F

minibatch = 128
gamma = 0.95

class SARSA_Agent():
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
        self.epsilon = 1.0 # Initialize epsilon inside agent or pass it in. 
                           # For consistency with DQN_Agent, we'll rely on get_action receiving epsilon, 
                           # but for learning we need it to select next_action.

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
            return np.random.choice(range(action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, gamma):
        # SARSA Update: Q(S, A) <- Q(S, A) + alpha * [R + gamma * Q(S', A') - Q(S, A)]
        # Where A' is chosen using the current policy (epsilon-greedy)
        
        states, actions, rewards, next_states, dones = experiences
        
        # Get next actions A' using the local network and current epsilon strategy
        # Note: We need an epsilon for the update. Since we don't store epsilon in experience,
        # we'll assume a small epsilon or use the current training epsilon if we had access to it.
        # For Deep SARSA, often we just use the greedy action (Q-Learning) or sample.
        # If we use greedy, it becomes Q-Learning (DQN).
        # To make it SARSA, we should sample A' from the policy.
        # Since we don't have the exact epsilon from when the data was collected (off-policy issue),
        # we can use Expected SARSA which averages over probabilities, OR
        # just select A' greedily (which is DQN).
        # BUT, the requirement is SARSA.
        # Let's implement "Deep SARSA" where we select A' based on the *current* policy.
        # We will assume a small epsilon for stability, e.g., 0.01 or just use argmax (which makes it DQN...).
        # Wait, if I use argmax it IS DQN.
        # To distinguish, I should probably use the actual next action if it was stored (On-Policy).
        # But ReplayMemory stores (s, a, r, s', done). It does NOT store next_action.
        # Standard DQN Replay Memory doesn't store next_action.
        # If I want true on-policy SARSA, I shouldn't use a large Replay Memory, or I should store (s,a,r,s',a').
        
        # DECISION: I will modify the learn step to calculate Expected SARSA value.
        # Expected SARSA: Target = R + gamma * sum(pi(a'|s') * Q_target(s', a'))
        # This is robust and works with Replay Memory.
        
        # Calculate Q values for next states
        with torch.no_grad():
            next_q_values = self.target_qnetwork(next_states) # [batch, action_size]
            
            # For Expected SARSA, we need probabilities of each action.
            # Let's assume a small epsilon for the "expectation".
            # Or simpler: Just use the value of the action that WOULD be selected.
            # But that's what I said above.
            
            # Let's stick to a simpler interpretation for this project:
            # Select next action A' based on local_network (with some epsilon noise or just greedy)
            # Then evaluate Q_target(S', A').
            # This is "Double DQN" style but without the argmax on target if we add noise.
            
            # Actually, let's implement strictly what is often called "Deep SARSA":
            # Q_target = R + gamma * Q_target(s', argmax(Q_local(s'))) 
            # This is actually Double DQN.
            
            # Let's go with:
            # Q_target = R + gamma * Q_target(s', a') 
            # where a' ~ EpsilonGreedy(Q_local(s'))
            
            # We need an epsilon. I'll use a fixed small epsilon for the update target, e.g. 0.02
            # This distinguishes it from DQN which is always greedy (epsilon=0) for the target.
            
            current_epsilon = 0.05 # Fixed small epsilon for target calculation
            
            best_actions = self.local_qnetwork(next_states).argmax(1).unsqueeze(1)
            
            # This is getting complicated to vectorize for epsilon greedy expectation.
            # Let's use the "Expected Value" directly.
            # E[Q] = (1-eps) * max(Q) + eps * mean(Q)
            
            max_next_q = next_q_values.max(dim=1)[0].unsqueeze(1)
            mean_next_q = next_q_values.mean(dim=1).unsqueeze(1)
            
            expected_next_q = (1 - current_epsilon) * max_next_q + current_epsilon * mean_next_q
            
            next_q_targets = expected_next_q

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

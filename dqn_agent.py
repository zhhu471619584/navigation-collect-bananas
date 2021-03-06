import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, buffer_size, batch_size, lr, tau,sequential_sampling_fre):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            buffer_size(int):replay buffer size
            batch_size(int): minibatch size
            lr(float):learning rate 
            tau(float):for soft update of target parameters
            sequential_sampling_fre(int):Ratio of random sampling to sequential sampling
            
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.tau = tau
        

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed, sequential_sampling_fre)

    
    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get max predicted Q values (for next states) from target model
        Q_local_argmax = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        Q_targets_next_states = self.qnetwork_target(next_states).detach().gather(1, Q_local_argmax)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next_states * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, sequential_sampling_fre):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            sequential_sampling_fre(int):Ratio of random sampling to sequential sampling
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.step = 0
        self.sequential_sampling_fre = sequential_sampling_fre
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
        
    
    def sample(self):
        """Random sampling or sequential sampling a batch of experiences from memory."""
        self.step = (self.step + 1) % self.sequential_sampling_fre
        
        if self.step == 0:
            experiences = np.arange(len(self.memory))[-self.batch_size:-1]
        else:
            experiences = np.random.choice(np.arange(len(self.memory)), size=self.batch_size,replace=False)
            
            
        states = torch.from_numpy(np.vstack([self.memory[i].state for i in experiences if i is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[i].action for i in experiences if i is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[i].reward for i in experiences if i is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[i].next_state for i in experiences if i is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[i].done for i in experiences if i is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
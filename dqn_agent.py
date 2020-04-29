import numpy as np
import random
from collections import namedtuple, deque
from model import QNetwork
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.optim as optim
from prioritized_memory import Memory



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
        self.step = 0
        self.sequential_sampling_fre = sequential_sampling_fre

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr)

        # Replay memory
        self.memory = Memory(self.buffer_size)
    
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

    def learn(self,gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, idxs, is_weights = self.sample()
        
        # Get max predicted Q values (for next states) from target model
        
        with torch.no_grad():
            Q_local_argmax = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
            Q_targets_next_states = self.qnetwork_target(next_states).detach().gather(1, Q_local_argmax)
            # Compute Q targets for current states 
            Q_targets = rewards + (gamma * Q_targets_next_states * (1 - dones))
        
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        errors = torch.abs(Q_expected-Q_targets).cpu().data.numpy().squeeze()
        # update priority
        for i in range(self.batch_size):
            idx = idxs[i]
            self.memory.update(idx, errors[i])
        
        # Compute loss
        self.optimizer.zero_grad()
        loss = (torch.FloatTensor(is_weights).to(device) * F.mse_loss(Q_expected, Q_targets)).mean()
        
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update()                     

    def soft_update(self):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)


    def append_sample(self, state, action, reward, next_state, done, gamma):
        with torch.no_grad():
            target = self.qnetwork_local(Variable(torch.FloatTensor(state).unsqueeze(0).to(device))).data
            old_val = target[0][action]
            target_val = self.qnetwork_target(Variable(torch.FloatTensor(next_state).unsqueeze(0).to(device))).data
            if done:
                target[0][action] = reward
            else:
                target[0][action] = reward + gamma * torch.max(target_val)

            error = abs(old_val - target[0][action])

            self.memory.add(error, (state, action, reward, next_state, done))
            
    def sample(self):
        """Random sampling or sequential sampling a batch of experiences from memory."""
        self.step = (self.step + 1) % self.sequential_sampling_fre
        if self.step == 0:
            mini_batch, idxs, is_weights = self.memory.sequential_sample(self.batch_size)
        else:
            mini_batch, idxs, is_weights = self.memory.sample(self.batch_size)
            
        mini_batch = np.array(mini_batch).transpose()
        states = torch.Tensor(np.vstack(mini_batch[0])).float().to(device)
        actions = torch.Tensor(np.vstack(mini_batch[1])).long().to(device)
        rewards = torch.FloatTensor(np.vstack(mini_batch[2])).to(device)
        next_states = torch.Tensor(np.vstack(mini_batch[3])).float().to(device)
        dones = torch.FloatTensor(np.vstack(mini_batch[4].astype(np.uint8))).to(device)
  
        return (states, actions, rewards, next_states, dones, idxs, is_weights)
       
    
            
     
import math
import random
import numpy as np
from collections import namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import DDQNModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class DQAgent:
    
    def __init__(self,
                 state_size,
                 action_size,
                 batch_size=128,
                 replay_capacity=1000, 
                 gamma=.99, 
                 alpha=.2, 
                 eps=1.0, 
                 eps_decay=.9999,
                 eps_min=.02,
                 target_update_interval=10,
                 learning_rate=1e-3,
                 lr_decay=.95):
        """
        Initialize agent and configure hyper-parameters
        """
        self.replay_mem = ReplayMemory(replay_capacity)
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.target_update_interval = target_update_interval
        self.target_q = DDQNModel(state_size, action_size)
        self.working_q = DDQNModel(state_size, action_size)
        self.state_size = state_size
        self.action_size = action_size
        self.n_steps = 0
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.optimizer = optim.Adam(self.working_q.parameters(),lr=learning_rate, weight_decay=1e-3)
    
    def sample_action(self, state):
        """
        Samples an action given input state vector
        """
        sample = random.random()
        eps_threshold = self.eps_min + (self.eps - self.eps_min) * \
            math.exp(-1. * self.n_steps / self.eps_decay)
        self.n_steps += 1
        state = torch.from_numpy(state).float().view(1,-1)
        self.working_q.eval()
        if sample > eps_threshold:
            with torch.no_grad():
                action = self.working_q(state).max(1)[1].view(1, 1).to(device=device)
        else:
            action = torch.tensor([[random.randrange(2)]], device=device, dtype=torch.long)
        self.working_q.train()
        return action
    
    def act(self, state):
        """
        Conduct an action given input state vector
        Used in eveluation when epsilon-greedy is not required
        """
        state = torch.from_numpy(state).float().view(1,-1)
        self.target_q.eval()
        action = self.target_q(state).to(device=device).max(1)[1].item()
        return action
        
    def update_model(self,*args):
        """
        Input is a tuple: ('state', 'action', 'next_state', 'reward')
        """
        self.replay_mem.push(*args)
        if not self.replay_mem.is_ready():
            return
        
        transitions = self.replay_mem.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.uint8)
        
        non_final_next_states = torch.cat([torch.from_numpy(s).float().view(1,-1) for s in batch.next_state if s is not None], dim=0)
        state_batch = torch.cat([torch.FloatTensor([a_state], device=device) for a_state in batch.state], dim=0)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat([torch.FloatTensor([[a_reward]], device=device) for a_reward in batch.reward])

        # Compute Q(s_t, a) : the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.working_q(state_batch).to(device=device).gather(1, action_batch)

        # Prepare memory for computing V(s_{t+1}) for all next states.
        next_state_values = torch.zeros([self.batch_size,1], device=device)
        
        # For Double Q Learning purpose, the columns are greedily selected
        # from the frequently updated Qnetwork instead of the Target QNetwork(TQN),
        # while the action-state values are selected from the TQN
        non_final_next_state_actions = self.working_q(non_final_next_states).to(device=device).max(1)[1].view(-1,1)
        self.target_q.eval()
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_q(non_final_next_states).to(device=device).gather(1, non_final_next_state_actions).detach()
        self.target_q.train()
        
        # Compute expected values
        expected_state_action_values = (next_state_values * torch.FloatTensor([self.gamma]).to(device=device)) + reward_batch.to(device=device)
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Constrain gradient from exploding
        for param in self.working_q.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.update_target_net()
        
    def save(self, path="./trained_model.checkpoint"):
        """
        Save state_dict of the target Q
        """
        torch.save({"state_dict":self.target_q.state_dict}, path)
        
    def load(self, path):
        """
        Load models and 
        """
        state_dict = torch.load(path)['state_dict']
        self.working_q.load_state_dict(state_dict())
        self.target_q.load_state_dict(state_dict())
        
        # If recoverred for training, decay learning rate
        self.learning_rate *= self.lr_decay
        for group in self.optimizer.param_groups:
            group['lr'] = self.learning_rate
        
    def update_target_net(self):
        """
        Periodically update the target QNetwork
        """
        if self.n_steps % self.target_update_interval == 0:
            self.target_q.load_state_dict(self.working_q.state_dict())
            

class ReplayMemory(object):
    """
    class wrapping replay memory functions
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """
        Saves a transition
        args: a tuple of records: ('state', 'action', 'next_state', 'reward')
        """        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def is_ready(self):
        return len(self.memory) == self.capacity


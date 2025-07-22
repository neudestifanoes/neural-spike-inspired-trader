import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQN(nn.Module): #This is the parent model, its a class that inherits from nn.module, and it predicts Q-values for each action, given a state, it is essential to understand this fundamentally, this tells u how good is it to take a certain action in a certain state?
    def __init__(self, state_dim, action_dim): # the inputs are purely dimensions, how many states and how many actions are available
        super(DQN, self).__init__()
        self.net = nn.Sequential(  #this is a container that would help us run the data in order when we call the forward method. 
            nn.Linear(state_dim, 64), #takes 2 values, spikes and entropy, and maps them to a 64d vector with different weights
            nn.ReLU(), #applies a non linearity, basically setting the negative values to zero, to allow complex patterns to be learned
            nn.Linear(64, 64), #map the 64d vector to another 64 vector, so each neuron takes in 64 inputs and applies 64 weights and one bias
            nn.ReLU(), # applies another non linearity to the new 64 d vector
            nn.Linear(64, action_dim) #maps the 64 d vector to the number of actions available, and these associated numberrs with these actions are their q values, how good are these actions to take given the very first state input.
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        self.model = DQN(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=10000)
        self.batch_size = 64

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad(): # we are telling py torch not to track gradients here because it is only evaluating and not training yet
            state_tensor = torch.FloatTensor(state).unsqueeze(0) #converting the input state to Float tensor adn unsquezz adds a batch dimention so it kind of becomes something like [1, state_dim]
            q_values = self.model(state_tensor) 
            return torch.argmax(q_values).item()

    def store(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor([np.array(s, dtype=np.float32) for s in states])
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor([float(r) for r in rewards]).unsqueeze(1)
        next_states = torch.FloatTensor([np.array(s, dtype=np.float32) for s in next_states])
        dones = torch.BoolTensor(dones).unsqueeze(1)

        q_values = self.model(states).gather(1, actions)
        next_q_values = self.model(next_states).max(1, keepdim=True)[0].detach()
        targets = rewards + self.gamma * next_q_values * (~dones)

        loss = self.criterion(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    
    def save_model(self, filepath):
        #saving the updated weight, all of them, to the file which I will specify in main
        torch.save(self.model.state_dict(), filepath)

    def load_model(self, filepath):
        # Loading the weight to use them as evaluation markers on a new unseen data, same stocks ofc
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()  # Set to evaluation mode

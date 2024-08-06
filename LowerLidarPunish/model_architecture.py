import math
import random
from collections import deque
#import airsim
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from setuptools import glob
#from env import DroneEnv
from torch.utils.tensorboard import SummaryWriter
import time
#from prioritized_memory import Memory

writer = SummaryWriter()

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Define the attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        attention_scores = torch.tanh(self.attention(lstm_output))
        attention_weights = torch.softmax(self.context_vector(attention_scores), dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights

# Define the neural network with LSTM, skip connections, and attention
class DQN(nn.Module):
    def __init__(self, in_channels=1, num_actions=6, hidden_dim=128):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 84, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(84, 42, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(42, 21, kernel_size=2, stride=2)
        self.fc4 = nn.Linear(21 * 4 * 4, 168)
        self.lstm = nn.LSTM(168, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, hidden_state):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        
        # Skip connections
        x3 += x2
        x2 += x1
        x3 = x3.view(batch_size, seq_len, -1)
        
        x3 = F.relu(self.fc4(x3))
        lstm_out, hidden_state = self.lstm(x3, hidden_state)
        
        context_vector, attention_weights = self.attention(lstm_out)
        
        return self.fc5(context_vector), hidden_state



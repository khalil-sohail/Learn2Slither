# this is the main file for the DQN agent.

from app_config.config import GLOBAL_CONFIG

import os
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim

# from model import LinearQNet  # Assuming you build the network in a separate file

class DQNAgent:
    def __init__(self):
        """
        Initializes hyperparameters from GLOBAL_CONFIG, sets up the Replay Memory, 
        and instantiates BOTH the Policy Network and the Target Network.
        """
        self.gamma = GLOBAL_CONFIG.get("gamma")
        self.epsilon_decay = GLOBAL_CONFIG.get("epsilon_decay")
        self.epsilon = 1.0 # Starts at 100% exploration
        self.epsilon_min = 0.01
        
        self.memory = deque(maxlen=GLOBAL_CONFIG.get("memory_capacity"))

    def get_action(self, state):
        """
        Implements the Epsilon-Greedy policy.
        Generates a random number: if it's less than epsilon, return a random move.
        Otherwise, pass the state through the policy_net and return the action with the highest Q-value.
        """
        pass

    def remember(self, state, action, reward, next_state, done):
        """
        Appends the (state, action, reward, next_state, done) tuple to the replay memory deque.
        """
        pass

    def train_step(self, batch_size):
        """
        1. Samples a random mini-batch from the replay memory.
        2. Calculates current Q-values using the policy_net.
        3. Calculates expected Q-values using the target_net and the Bellman equation.
        4. Computes the loss, performs backpropagation, and steps the optimizer.
        """
        pass

    def update_target_network(self):
        """
        Copies the weights from the policy_net to the target_net.
        Called every few episodes to stabilize the training process.
        """
        pass

    def save_model(self, file_name="model.pth"):
        """
        Serializes and saves the policy_net weights to the disk using torch.save().
        """
        pass

    def load_model(self, file_path):
        """
        Loads pre-trained weights from the disk into both networks for inference or resumed training.
        """
        pass
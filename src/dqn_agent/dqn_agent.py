# this is the main file for the DQN agent.

from app_config.config import GLOBAL_CONFIG

import os
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from constants import Action, Entity
from collections import deque
import numpy as np

class DQNAgent(nn.Module):
    def __init__(self):
        """
        Initializes hyperparameters from GLOBAL_CONFIG, sets up the Replay Memory, 
        and instantiates BOTH the Policy Network and the Target Network.
        """
        super(DQNAgent, self).__init__()
        self.memory         = deque(maxlen=GLOBAL_CONFIG.get("memory_capacity"))
        self.gamma          = GLOBAL_CONFIG.get("gamma")                # discount factor for future rewards
        self.learning_rate  = GLOBAL_CONFIG.get("learning_rate")        # learning rate for the optimizer
        self.epsilon_decay  = GLOBAL_CONFIG.get("epsilon_decay")        # exploration rate for epsilon-greedy policy
        self.epsilon        = 1                                         # initial exploration rate
        self.min_epsilon    = 0.01

        self.q_network = nn.Sequential(
            nn.Linear(16, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )

    def forward(self, x):
        """
        Defines the forward pass of the network.
        :param x: Input tensor (state representation).
        :return: Output tensor (Q-values for each action).
        """
        return self.q_network(x)

    def get_action(self, snake_vision, exploit_only=True):
        """
        Chooses an action based on the epsilon-greedy policy:
        - With probability epsilon, choose a random action (exploration).
        - With probability (1 - epsilon), choose the action that maximizes the Q-value (exploitation).
        param snake_vision: A 16-value float32 NumPy array representing the snake's vision.
        return: The action to be taken (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT).
        """
        if not exploit_only and random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            state_tensor = torch.FloatTensor(snake_vision).unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor) 
            
            return torch.argmax(q_values).item()


    def remember(self, snake_vision, action, next_snake_vision, reward, done):
        """
        Appends the (state, action, reward, next_state, done) tuple to the replay memory deque.
        """
        self.memory.append((
            snake_vision,
            action,
            next_snake_vision,
            reward,
            done
        ))

    def train_step(self, batch_size):
        """
        Trains the neural network by sampling a batch of past experiences
        and applying the Q-learning update rule.
        """
        # 1. Check if we have enough memory
        # We cannot train on a batch of 64 if the snake has only taken 10 steps!
        if len(self.memory) < batch_size:
            return

        # 2. Sample a random batch of experiences from the deque
        batch = random.sample(self.memory, batch_size)
        
        # Unpack the batch into separate tuples
        snake_vision, action, next_snake_vision, reward, done = zip(*batch)

        # 3. Convert data into PyTorch Tensors
        # Neural networks require data to be in tensor format to do math and backpropagation
        states = torch.from_numpy(np.array(snake_vision, dtype=np.float32))
        next_states = torch.from_numpy(np.array(next_snake_vision, dtype=np.float32))

        
        actions = torch.tensor(action, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(reward, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(done, dtype=torch.float32).unsqueeze(1)

        # 4. Get the Current Q-Values
        # Pass the batch of states through the single Q-Network.
        # .gather(1, actions) extracts only the Q-value for the specific action the snake actually took.
        current_q = self.q_network(states).gather(1, actions)

        # 5. Compute the Target Q-Values (The Bellman Equation)
        with torch.no_grad():
            # Get the highest predicted Q-value for the next state
            max_next_q = self.q_network(next_states).amax(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # 6. Compute Loss and Backpropagate
        # Measure the error between what the network guessed and the new target
        loss = self.criterion(current_q, target_q)

        before = self.q_network[0].weight.detach().clone()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        after = self.q_network[0].weight.detach()

        # print(torch.allclose(before, after))

        self.epsilon = max(
            self.min_epsilon,
            self.epsilon * self.epsilon_decay
        )

        return loss.item()

    def save_model(self, file_name="./model.pth"):
        """
        Serializes and saves the policy_net weights to the disk using torch.save().
        """
        torch.save(self.state_dict(), file_name)

    def load_model(self, file_path):
        """
        Loads pre-trained weights from the disk into both networks for inference or resumed training.
        """
        pass
        # state_dict = torch.load(file_path, map_location=torch.device("cpu"))
        # self.load_state_dict(state_dict)
        # self.eval()

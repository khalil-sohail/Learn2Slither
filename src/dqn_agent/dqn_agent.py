# this is the main file for the DQN agent.

from app_config.config import GLOBAL_CONFIG
from constants import Action, AppState

import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class DQNAgent(nn.Module):
    def __init__(self, learning_enabled=True):
        """
        Initializes hyperparameters from GLOBAL_CONFIG, sets up the Replay Memory,
        and instantiates the Q-Network used to approximate the Q function.

        :param learning_enabled: When False, the agent behaves in pure
            exploitation / evaluation mode: it will not explore randomly,
            will not store transitions in replay memory, and train_step()
            becomes a no-op. This lets you evaluate a model's success rate
            without altering its learning state (subject requirement:
            "Exploitation without learning").
        """
        super(DQNAgent, self).__init__()
        self.memory         = deque(maxlen=GLOBAL_CONFIG.get("memory_capacity"))
        self.epsilon        = 1
        self.min_epsilon    = 0.05
        self.losses         = []
        self.learning_enabled = learning_enabled
        torch.backends.cudnn.benchmark = True
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.q_network = nn.Sequential(
            nn.Linear(16, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )
        
        # self.target_network = nn.Sequential(
        #     nn.Linear(16, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 4),
        # )

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=GLOBAL_CONFIG.get("learning_rate")
        )

        self.to(self.device)
        self.load_model()

    def forward(self, x):
        """
        Defines the forward pass of the network.
        :param x: Input tensor (state representation).
        :return: Output tensor (Q-values for each action).
        """
        return self.q_network(x.to(self.device))

    def get_action(self, snake_vision, explore_mode=False) -> Action:
        """
        Chooses an action based on the epsilon-greedy policy:
        - With probability epsilon, choose a random action (exploration).
        - With probability (1 - epsilon), choose the action that maximizes the Q-value (exploitation).

        If self.learning_enabled is False, exploration is always disabled
        regardless of exploit_only, since an evaluation-only agent should
        never take random actions.

        :param snake_vision: A 16-value float32 NumPy array representing the snake's vision.
        :param explore_mode: If True, allows for exploration (random actions).
        :return: The action index to be taken (0-3, corresponding to
            Action.UP / Action.DOWN / Action.LEFT / Action.RIGHT).
        """
        explore = explore_mode and self.learning_enabled

        if explore and random.random() < self.epsilon:
            return Action(random.randint(0, 3))
        else:
            state_tensor = torch.FloatTensor(snake_vision).unsqueeze(0)

            with torch.no_grad():
                q_values = self.q_network(state_tensor.to(self.device))

            return Action(torch.argmax(q_values).item())

    def remember(self, snake_vision, action, next_snake_vision, reward, done):
        """
        Appends the (state, action, reward, next_state, done) tuple to the replay memory deque.
        No-op when learning is disabled, so evaluation runs never mutate
        the agent's learning state.
        """
        if not self.learning_enabled:
            return

        self.memory.append((
            snake_vision,
            action,
            next_snake_vision,
            reward,
            done
        ))

    def train_step(self, batch_size, update_epsilon: bool = True):
        """
        Trains the neural network by sampling a batch of past experiences
        and applying the Q-learning update rule.
        No-op when learning is disabled (evaluation / exploitation-only mode).
        """
        if not self.learning_enabled:
            return

        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        snake_vision, action, next_snake_vision, reward, done = zip(*batch)
        states = torch.from_numpy(
            np.array(snake_vision, dtype=np.float32)
        ).to(self.device)

        next_states = torch.from_numpy(
            np.array(next_snake_vision, dtype=np.float32)
        ).to(self.device)

        actions = torch.tensor(
            action,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            reward,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        dones = torch.tensor(
            done,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(1)

        current_q = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            max_next_q = self.q_network(next_states).amax(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * GLOBAL_CONFIG.get("gamma") * max_next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        self.optimizer.step()

        if self.epsilon > self.min_epsilon and update_epsilon:
            self.epsilon = max(
                self.min_epsilon,
                self.epsilon * GLOBAL_CONFIG.get("epsilon_decay")
            )

        # with torch.no_grad():
        #     q = self.q_network(states)
        # print(
        #     q.min().item(),
        #     q.max().item(),
        #     q.mean().item(),
        # )

        self.losses.append(loss.item())
        return loss.item()

    def save_model(self):
        """
        Serializes and saves everything describing the agent's current
        learning state to disk: network weights, optimizer state, and
        epsilon. This is what makes a saved file a true "model" that a
        new training session can resume from exactly where the last one
        left off, per the subject's export/import requirement.
        """
        if GLOBAL_CONFIG.get("new_weights") is not None:
            torch.save({
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "memory": self.memory,
            }, GLOBAL_CONFIG.get("new_weights"))

    def load_model(self):
        """
        Loads a previously saved learning state (network weights, optimizer
        state, and epsilon) from disk, restoring the agent to exactly the
        point it was saved at so training or evaluation can continue.
        """
        if GLOBAL_CONFIG.get("target_weights") is not None:
            checkpoint = torch.load(
                GLOBAL_CONFIG.get("target_weights"),
                map_location=self.device,
                weights_only=False
            )
            self.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epsilon = checkpoint["epsilon"]
            self.memory = checkpoint["memory"]

    def __del__(self):
        """
        Ensures that the optimizer's state is cleared when the agent is
        deleted, preventing memory leaks in long-running training sessions.
        """
        self.save_model()

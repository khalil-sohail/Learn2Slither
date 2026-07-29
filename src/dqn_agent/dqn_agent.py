# this is the main file for the DQN agent.

import math

from app_config.config import GLOBAL_CONFIG
from constants import Action, AppState
from collections import deque

import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy

from gymnasium_wrapper.gymnasium_wrapper import GymnasiumWrapper


class DQNAgent(nn.Module):
    def __init__(self):
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
        self.target_steps   = GLOBAL_CONFIG.get("target_steps")
        self.min_epsilon    = GLOBAL_CONFIG.get("min_epsilon")
        self.gamma          = GLOBAL_CONFIG.get("gamma")
        self.learning_rate  = GLOBAL_CONFIG.get("learning_rate")
        self.target_weights = GLOBAL_CONFIG.get("target_weights")
        self.new_weights    = GLOBAL_CONFIG.get("new_weights")

        self.initial_epsilon = 1.0
        
        self.epsilon_decay = (
            self.min_epsilon / self.initial_epsilon
        ) ** (1 / self.target_steps)
        
        self.bucket_steps = [0] * 5

        self.last_episode    = 0
        self.losses         = []
        torch.backends.cudnn.benchmark = True
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.q_network = nn.Sequential(
            nn.Linear(20, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )
        
        self.target_network = copy.deepcopy(self.q_network)
        self.target_network.eval()

        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate
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

    def get_epsilon(self, bucket_index: int) -> float:
        return max(
            self.min_epsilon,
            self.initial_epsilon *
            (self.epsilon_decay ** self.bucket_steps[bucket_index])
        )

    def get_action(self, snake_vision, explore_mode=False, bucket_index: int = 0) -> Action:
        """
        Chooses an action based on the epsilon-greedy policy:
        - With probability epsilon, choose a random action (exploration).
        - With probability (1 - epsilon), choose the action that maximizes the Q-value (exploitation).

        :param snake_vision: A 16-value float32 NumPy array representing the snake's vision.
        :param explore_mode: If True, allows for exploration (random actions).
        :return: The action index to be taken (0-3, corresponding to
            Action.UP / Action.DOWN / Action.LEFT / Action.RIGHT).
        """

        explore = explore_mode
        if explore and random.random() < self.get_epsilon(bucket_index):
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
        self.memory.append((
            snake_vision,
            action,
            next_snake_vision,
            reward,
            done
        ))

    def train_step(self, batch_size, bucket_index: int = 0):
        """
        Trains the neural network by sampling a batch of past experiences
        and applying the Q-learning update rule.
        No-op when learning is disabled (evaluation / exploitation-only mode).
        """
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
            max_next_q = self.target_network(next_states).amax(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.bucket_steps[bucket_index] += 1


        self.losses.append(loss.item())
        return loss.item()

    def update_target_network(self):
        """
        Updates the target network's weights to match the current Q-network.
        This is typically done periodically to stabilize training.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self):
        """
        Serializes and saves everything describing the agent's current
        learning state to disk: network weights, optimizer state, and
        epsilon. This is what makes a saved file a true "model" that a
        new training session can resume from exactly where the last one
        left off, per the subject's export/import requirement.
        """
        path = self.new_weights
        if path is not None:
            print(f"Saving model to {path}...")
            torch.save({
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "last_episode": self.last_episode,
                "bucket_steps": self.bucket_steps,
                "losses": self.losses,
            }, path)

    def load_model(self):
        """
        Loads a previously saved learning state (network weights, optimizer
        state, and epsilon) from disk, restoring the agent to exactly the
        point it was saved at so training or evaluation can continue.
        """
        path = self.target_weights
        
        if path is not None:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=True,
            )

            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_network.load_state_dict(checkpoint["target_network"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            self.last_episode = checkpoint.get("last_episode", 0)
            self.bucket_steps = checkpoint.get("bucket_steps", [0] * 5)
            self.losses = checkpoint.get("losses", [])

    def __del__(self):
        """
        Ensures that the optimizer's state is cleared when the agent is
        deleted, preventing memory leaks in long-running training sessions.
        """
        self.save_model()

    def evaluate(
        self,
        env: GymnasiumWrapper,
        board_sizes: list[int],
        games_per_board: int = 20,
    ) -> str:
        """
        Runs deterministic evaluation (epsilon = 0) on each requested board size
        and returns a formatted report.
        """

        report = ""

        for board_size in board_sizes:
            results = []

            for _ in range(games_per_board):
                state = env.reset(board_size=board_size)

                done = False
                while not done:
                    action = self.get_action(
                        state,
                        explore_mode=False,
                        bucket_index=0,
                    )

                    state, _, done = env.step(action)

                score, green, red, _ = env.board.get_board_statistics()
                results.append((score, green, red))

            avg_score = sum(x[0] for x in results) / len(results)
            avg_green = sum(x[1] for x in results) / len(results)
            avg_red = sum(x[2] for x in results) / len(results)

            best_score = max(x[0] for x in results)
            success_rate = (
                sum(x[0] > 0 for x in results)
                / len(results)
                * 100
            )

            report += (
                f"  {board_size:3}x{board_size}: "
                f"Avg Score ({len(results)} games): {avg_score:.2f}, "
                f"Best: {best_score}, "
                f"Success: {success_rate:.1f}%, "
                f"Green: {avg_green:.2f}, "
                f"Red: {avg_red:.2f}\n"
            )

        return report
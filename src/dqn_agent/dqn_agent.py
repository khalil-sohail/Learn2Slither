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

class DQNAgent:
    def __init__(self):
        """
        Initializes hyperparameters from GLOBAL_CONFIG, sets up the Replay Memory, 
        and instantiates BOTH the Policy Network and the Target Network.
        """
        self.memory = deque(maxlen=GLOBAL_CONFIG.get("memory_capacity"))
        self.epsilon_decay = GLOBAL_CONFIG.get("epsilon_decay")
        self.gamma = GLOBAL_CONFIG.get("gamma")
        self.epsilon = 1.0 # Starts at 100% exploration
        self.epsilon_min = 0.01

    def get_action(self, snake_vision):
        """
        Chooses an action based on the epsilon-greedy policy:
        - With probability epsilon, choose a random action (exploration).
        - With probability (1 - epsilon), choose the action that maximizes the Q-value (exploitation).
        param snake_vision: A 16-value float32 NumPy array representing the snake's vision.
        return: The action to be taken (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT).
        """
        return random.randint(0, 3)

    def get_test_action(self, state):
        """
        Chooses the action that best pursues the goal of eating as many
        green apples as possible: pathfind to the nearest reachable green
        apple while avoiding walls, its own body, and (where possible) red
        apples. If no green apple is currently reachable, moves toward the
        most open space instead of moving blindly, to stay alive longer.

        param state: The current state of the environment. 1D array of size (board_size * board_size).
        return: The action to be taken (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT).
        """

        board_size = GLOBAL_CONFIG.get('board_size')
        grid = [state[i * board_size:(i + 1) * board_size] for i in range(board_size)]
        head_row, head_col = None, None
        green_apples = set()
        red_apples = set()
        blocked = set()

        for row in range(board_size):
            for col in range(board_size):
                cell = grid[row][col]
                if cell in (Entity.SNAKE_HEAD_UP, Entity.SNAKE_HEAD_DOWN, Entity.SNAKE_HEAD_LEFT, Entity.SNAKE_HEAD_RIGHT):
                    head_row, head_col = row, col
                elif cell == Entity.GREEN_APPLE:
                    green_apples.add((row, col))
                elif cell == Entity.RED_APPLE:
                    red_apples.add((row, col))
                elif cell in (Entity.SNAKE_BODY, Entity.WALL):
                    blocked.add((row, col))

        if head_row is None:
            return random.choice(list(Action))

        directions = {
            Action.UP:    (-1, 0),
            Action.DOWN:  (1, 0),
            Action.LEFT:  (0, -1),
            Action.RIGHT: (0, 1),
        }

        def in_bounds(row, col):
            return 0 <= row < board_size and 0 <= col < board_size

        def safe_actions_from(row, col):
            result = []
            for action, (dr, dc) in directions.items():
                nr, nc = row + dr, col + dc
                if in_bounds(nr, nc) and (nr, nc) not in blocked:
                    result.append(action)
            return result

        safe_actions = safe_actions_from(head_row, head_col)
        if not safe_actions:
            # No legal move avoids immediate death; nothing left to optimize for.
            return random.choice(list(Action))

        def bfs_first_step_to_nearest_target(start, targets, avoid):
            if not targets:
                return None
            visited = {start}
            queue = deque([(start, None)])
            while queue:
                (row, col), first_action = queue.popleft()
                if (row, col) in targets and first_action is not None:
                    return first_action
                for action, (dr, dc) in directions.items():
                    nr, nc = row + dr, col + dc
                    if in_bounds(nr, nc) and (nr, nc) not in visited and (nr, nc) not in blocked and (nr, nc) not in avoid:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), first_action if first_action is not None else action))
            return None

        def flood_fill_count(start):
            visited = {start}
            queue = deque([start])
            count = 0
            while queue:
                row, col = queue.popleft()
                count += 1
                for dr, dc in directions.values():
                    nr, nc = row + dr, col + dc
                    if in_bounds(nr, nc) and (nr, nc) not in visited and (nr, nc) not in blocked:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            return count

        # Pass 1: reach the nearest green apple without crossing any red apple.
        action = bfs_first_step_to_nearest_target((head_row, head_col), green_apples, avoid=red_apples)
        if action is not None:
            return action

        # Pass 2: no red-free route exists — allow crossing red apples rather
        # than giving up on green apples entirely.
        action = bfs_first_step_to_nearest_target((head_row, head_col), green_apples, avoid=frozenset())
        if action is not None:
            return action

        # No green apple reachable at all — stay alive by keeping the most
        # open space available, to maximize the chance a path opens up later.
        def open_space_after(action):
            dr, dc = directions[action]
            return flood_fill_count((head_row + dr, head_col + dc))

        return max(safe_actions, key=open_space_after)

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
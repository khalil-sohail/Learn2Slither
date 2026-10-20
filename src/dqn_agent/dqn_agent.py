import os

from gymnasium_wrapper.gymnasium_wrapper import GymnasiumWrapper
from utilize.get_bucket_index import BOARD_BUCKETS
from .memory import ReplayBuffer, Transition
from app_config.config import GLOBAL_CONFIG
from torch.backends import cudnn
from collections import deque
from constants import Action

import torch.optim as optim
import torch.cuda as cuda
import torch.nn as nn
import numpy as np
import random
import torch
import copy


class DQNAgent(nn.Module):
    def __init__(self):
        """
        """
        super(DQNAgent, self).__init__()
        
        self.learning_rate      = GLOBAL_CONFIG.get("learning_rate")
        self.gamma              = GLOBAL_CONFIG.get("gamma")
        self.end_total_steps    = GLOBAL_CONFIG.get("end_total_steps")
        self.target_weights     = GLOBAL_CONFIG.get("target_weights")
        self.new_weights        = GLOBAL_CONFIG.get("new_weights")

        self.memory_capacity    = 200_000
        self.memory             = [ReplayBuffer(self.memory_capacity) for _ in range(5)]
        self.min_epsilon        = 0.01
        self.initial_epsilon    = 1.0
        
        self.total_steps        = 0
        self.env_steps          = 0

        self.last_episode       = 0
        self.losses             = []
        self.balancing_steps    = 0
        self.bucket_steps       = [0] * 5
        cudnn.benchmark         = True
        self.device             = torch.device("cuda" if cuda.is_available() else "cpu")

        self.q_network = nn.Sequential(
            nn.Linear(29, 32),
            nn.ReLU(),
            
            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 4),
        ).to(self.device)
        
        self.target_network = copy.deepcopy(self.q_network)
        self.target_network.eval()
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        self.load_model()
    
    def forward(self, x):
        """
        """
        return self.q_network(x.to(self.device))


    def get_epsilon(self, bucket_index: int) -> float:
        initial = max(0.6, self.initial_epsilon - bucket_index * 0.1)
        decay = (self.min_epsilon / initial) ** (1 / self.end_total_steps)

        return max(self.min_epsilon, initial * (decay ** self.bucket_steps[bucket_index]))

    def get_action(self, snake_vision, explore_mode=False, bucket_index: int = 0) -> Action:
        """
        """

        if explore_mode and random.random() < self.get_epsilon(bucket_index):
            return Action(random.randint(0, 3))
        else:
            state_tensor = torch.FloatTensor(snake_vision).unsqueeze(0)

            with torch.no_grad():
                q_values = self.q_network(state_tensor.to(self.device))

            return Action(torch.argmax(q_values).item())

    def remember(self, bucket_index: int, snake_vision: np.ndarray, action: Action, next_snake_vision: np.ndarray, reward: float, done: bool):
        """
        """
        self.memory[bucket_index].append(
            Transition(
                state=snake_vision.copy(),
                action=action,
                next_state=next_snake_vision.copy(),
                reward=reward,
                done=done,
            )
        )

    def train_step(self, batch_size: int, bucket_index: int = 0, balancing_phase: bool = False) -> float | None:
        """
        """

        if len(self.memory[bucket_index]) < batch_size:
            return None

        if bucket_index == 0:
            batch = random.sample(self.memory[0], batch_size)
        else:
            bucket_progress = self.bucket_steps[bucket_index] / self.end_total_steps

            old_fraction = max(0.20, 0.30 * (1 - bucket_progress))
            old_size = round(batch_size * old_fraction)

            previous = list(range(bucket_index))
            available = [i for i in previous if len(self.memory[i]) > 0]
            old_batch = []

            if available:
                weights = np.array([i + 1 for i in available], dtype=float)
                weights /= weights.sum()

                counts = np.random.multinomial(old_size, weights)
                for bucket, count in zip(available, counts):
                    old_batch.extend(
                        random.sample(
                            self.memory[bucket],
                            min(count, len(self.memory[bucket]))
                        )
                    )

            current_size = batch_size - len(old_batch)
            batch = random.sample(self.memory[bucket_index], current_size)
            batch.extend(old_batch)

        random.shuffle(batch)
        states = torch.tensor(
            np.array([t.state for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.tensor(
            [t.action for t in batch],
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(1)
        next_states = torch.tensor(
            np.array([t.next_state for t in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        rewards = torch.tensor(
            [t.reward for t in batch],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)
        dones = torch.tensor(
            [t.done for t in batch],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(1)

        current_q = self.q_network(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_network(next_states).amax(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
        self.optimizer.step()

        if balancing_phase:
            self.balancing_steps += 1
        else:
            self.bucket_steps[bucket_index] += 1
        self.total_steps += 1

        self.losses.append(loss.item())
        return loss.item()

    def update_target_network(self):
        """
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, user_blocked_path: str = ""):
        """
        """
        path = user_blocked_path or self.new_weights
        if not path:
            return

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "last_episode": self.last_episode,
            "bucket_steps": self.bucket_steps,
            "losses": self.losses,
            "total_steps": self.total_steps,
            "env_steps": self.env_steps,
            "memory": [list(bucket) for bucket in self.memory],
        }, path)
        print(f"Saved checkpoint to {path}:")

    def load_model(self):
        """
        """
        path = self.target_weights
        
        if path is not None:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )

            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_network.load_state_dict(checkpoint["target_network"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            self.last_episode = checkpoint.get("last_episode", 0)
            self.bucket_steps = checkpoint.get("bucket_steps", [0] * 5)
            self.losses = checkpoint.get("losses", [])
            self.total_steps = checkpoint.get("total_steps", 0)
            self.memory = [
                ReplayBuffer(self.memory_capacity)
                for _ in range(5)
            ]

            saved_memory = checkpoint["memory"]
            if len(saved_memory) != len(self.memory):
                raise ValueError("Checkpoint bucket count mismatch")

            for buffer, transitions in zip(self.memory, saved_memory):
                for transition in transitions:
                    buffer.append(transition)

            print(f"Loaded checkpoint:")
            print(f"  episode      : {self.last_episode}")
            print(f"  total_steps  : {self.total_steps:,}")
            print(f"  bucket_steps : {self.bucket_steps}")
            print(f"  memory size  : {[len(self.memory[i]) for i in range(5)]}")
            print(f"  losses       : {len(self.losses)} entries")

    def evaluate(self, env: GymnasiumWrapper, games_per_board: int = 20) -> str:
        """
        """

        # BUCKETS = [
        #     [7, 10, 12, 14, 15],
        #     [16, 18, 20, 24, 30],
        #     [32, 35, 40, 45, 50],
        #     [55, 60, 65, 70, 75],
        #     [80, 85, 90, 95, 100],
        # ]

        BUCKETS = BOARD_BUCKETS

        MIN_BUCKET_STEPS = self.end_total_steps * 0.2

        report = ""
        buckets_scores = []
        buckets_greens = []
        buckets_reds = []
        buckets_successes = []
        for bucket_index, board_sizes in enumerate(BUCKETS):

            if self.bucket_steps[bucket_index] < MIN_BUCKET_STEPS:
                print(f"Skipping bucket {bucket_index + 1} (not enough training steps: {self.bucket_steps[bucket_index]:,})")
                continue

            bucket_scores = []
            bucket_greens = []
            bucket_reds = []
            bucket_successes = []

            report += (
                f"    Bucket {bucket_index + 1} "
                f"({self.bucket_steps[bucket_index]:,} steps, "
                f"Epsilon: {self.get_epsilon(bucket_index):.4f})\n"
            )

            for board_size in board_sizes:
                results = []

                for _ in range(games_per_board):
                    state = env.reset(board_size=board_size)

                    done = False
                    while not done:
                        action = self.get_action(
                            state,
                            explore_mode=False,
                        )

                        state, _, done = env.step(action)

                    score, green, red, _ = env.board.get_board_statistics()
                    results.append((score, green, red))

                scores = [x[0] for x in results]
                greens = [x[1] for x in results]
                reds = [x[2] for x in results]

                avg_score = np.mean(scores)
                median_score = np.median(scores)
                std_score = np.std(scores)

                avg_green = np.mean(greens)
                avg_red = np.mean(reds)

                best_score = max(scores)
                worst_score = min(scores)
                success_rate = np.mean(np.array(scores) > 0) * 100

                bucket_scores.append(avg_score)
                bucket_greens.append(avg_green)
                bucket_reds.append(avg_red)
                bucket_successes.append(success_rate)

                buckets_scores.append(avg_score)
                buckets_greens.append(avg_green)
                buckets_reds.append(avg_red)
                buckets_successes.append(success_rate)

                report += (
                    f"      {board_size:>3}x{board_size:<3} {games_per_board} "
                    f"| Avg: {avg_score:>6.2f} "
                    f"| Median: {median_score:>6.2f} "
                    f"| Std: {std_score:>6.2f} "
                    f"| Best: {best_score:>3} "
                    f"| Worst: {worst_score:>3} "
                    f"| Success Rate: {success_rate:>6.1f}% "
                    f"| Green: {avg_green:>6.2f} "
                    f"| Red: {avg_red:>6.2f}\n"
                )
            

            report += (
                "      -------------------------------------------------------------------------------\n"
                f"      Avg Score        : {np.mean(bucket_scores):>6.2f}\n"
                f"      Avg Success Rate : {np.mean(bucket_successes):>6.1f}%\n"
                f"      Avg Green        : {np.mean(bucket_greens):>6.2f}\n"
                f"      Avg Red          : {np.mean(bucket_reds):>6.2f}\n"
                "      -------------------------------------------------------------------------------\n\n"
            )
        
        if len(buckets_scores) == 0:
            report += "No buckets have been sufficiently trained for evaluation."
            return report
        report += (
            "          -------------------------------------------------------------------------------\n"
            f"          All Avg Score        : {np.mean(buckets_scores):>6.2f}\n"
            f"          All Avg Success Rate : {np.mean(buckets_successes):>6.1f}%\n"
            f"          All Avg Green        : {np.mean(buckets_greens):>6.2f}\n"
            f"          All Avg Red          : {np.mean(buckets_reds):>6.2f}\n"
            "          -------------------------------------------------------------------------------\n\n"
        )

        return report
    
    

import os

from gymnasium_wrapper.gymnasium_wrapper import GymnasiumWrapper
from utilize.get_bucket_index import BOARD_BUCKETS
from .memory import ReplayBuffer, Transition
from app_config.config import GLOBAL_CONFIG
from torch.backends import cudnn
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
        Initializes the DQNAgent with its neural network, optimizer,
            and other parameters.
        """
        super(DQNAgent, self).__init__()

        self.end_total_steps = GLOBAL_CONFIG.get("end_total_steps")
        self.loads_module = GLOBAL_CONFIG.get("loading_wieghts")
        self.learning_rate = GLOBAL_CONFIG.get("learning_rate")
        self.gamma = GLOBAL_CONFIG.get("gamma")

        self.bucket_count = len(BOARD_BUCKETS)
        self.memory_capacity = 500_000

        self.memory = [
            ReplayBuffer(self.memory_capacity)
            for _ in range(self.bucket_count)
        ]
        self.min_epsilon = 0.01

        self.initial_epsilon = [
            1.00, 0.85, 0.70,
            0.55, 0.40, 0.40,
            0.35, 0.30, 0.25,
            0.20
        ]
        self.epsilon_decay_power = [
            1.50, 1.75, 2.00,
            2.25, 2.50, 2.50,
            2.75, 3.00, 3.25,
            3.50
        ]

        self.total_steps = 0
        self.env_steps = 0

        self.last_episode = 0
        self.losses = []
        self.balancing_steps = 0
        self.bucket_steps = [0] * self.bucket_count

        self.session_episodes_completed = 0

        self.updates_per_bucket = [0] * self.bucket_count
        self.balancing_updates_per_bucket = [0] * self.bucket_count

        cudnn.benchmark = True
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")

        self.q_network = nn.Sequential(
            nn.Linear(33, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 4),
        ).to(self.device)

        self.target_network = copy.deepcopy(self.q_network)
        self.target_network.eval()
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=self.learning_rate
        )

        if self.loads_module is not None:
            self.load_model()

    def forward(self, x):
        """
        Performs a forward pass through the Q-network.
        """
        return self.q_network(x.to(self.device))

    def get_epsilon(self, bucket_index: int) -> float:
        """
        equation:
            epsilon = min_epsilon + (initial_epsilon - min_epsilon)
                        * (1 - progress) ** decay_power
        where:
            progress = bucket_steps / end_total_steps
        """
        training_sessions = GLOBAL_CONFIG.get("training_sessions")
        if training_sessions is None:
            assert len(self.epsilon_decay_power) == self.bucket_count
            assert len(self.initial_epsilon) == self.bucket_count

            initial = self.initial_epsilon[bucket_index]
            power = self.epsilon_decay_power[bucket_index]

            progress = min(
                1.0,
                self.bucket_steps[bucket_index] / self.end_total_steps
            )
            return (
                self.min_epsilon + (initial - self.min_epsilon)
                * (1.0 - progress) ** power
            )
        else:
            initial = 1.00
            power = 3.00
            progress = min(
                1.0,
                self.session_episodes_completed / training_sessions
            )
            return (
                self.min_epsilon + (initial - self.min_epsilon)
                * (1.0 - progress) ** power
            )

    def get_action(
        self,
        snake_vision,
        explore_mode=False,
        bucket_index: int = 0
    ) -> Action:
        """
        Selects an action based on the current state (snake_vision) and
        exploration strategy (epsilon-greedy).
        """

        if explore_mode and random.random() < self.get_epsilon(bucket_index):
            return Action(random.randint(0, 3))
        else:
            state_tensor = torch.FloatTensor(snake_vision).unsqueeze(0)

            with torch.no_grad():
                q_values = self.q_network(state_tensor.to(self.device))

            return Action(torch.argmax(q_values).item())

    def remember(
        self,
        bucket_index: int,
        snake_vision: np.ndarray,
        action: Action,
        next_snake_vision: np.ndarray,
        reward: float,
        done: bool
    ):
        """
        Stores a transition in the agent's memory.
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

    def sample_training_batch(self, batch_size: int, bucket_index: int):
        """
        Samples a batch of transitions from the agent's memory.
        """
        memory = self.memory[bucket_index]

        if len(memory) < batch_size:
            return None

        return random.sample(memory, batch_size)

    def train_step(
        self,
        batch_size: int,
        bucket_index: int = 0,
        balancing_phase: bool = False
    ) -> float | None:
        """
        Performs a single training step for the agent's Q-network.
        """

        if len(self.memory[bucket_index]) < batch_size:
            return None

        batch = self.sample_training_batch(batch_size, bucket_index)
        if batch is None:
            return None

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
            max_next_q = (
                self.target_network(next_states).amax(dim=1, keepdim=True)
            )
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            max_norm=5.0
        )
        self.optimizer.step()

        self.total_steps += 1
        self.updates_per_bucket[bucket_index] += 1
        if balancing_phase:
            self.balancing_steps += 1
            self.balancing_updates_per_bucket[bucket_index] += 1
        else:
            self.bucket_steps[bucket_index] += 1

        self.losses.append(loss.item())
        return loss.item()

    def update_target_network(self):
        """
        Updates the target network with the current Q-network parameters.
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, path: str):
        """
        Saves the model parameters to a checkpoint file.
        """
        if not path:
            return

        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),

            "last_episode": self.last_episode,
            "bucket_steps": self.bucket_steps,
            "balancing_steps": self.balancing_steps,
            "total_steps": self.total_steps,
            "env_steps": self.env_steps,
        }, path)
        print(f"Saved checkpoint to {path}:")

    def load_model(self):
        """
        Loads the model parameters from a checkpoint file.
        """
        path = self.loads_module

        if path is not None:
            checkpoint = torch.load(
                path,
                map_location=self.device,
                weights_only=False,
            )

            self.q_network.load_state_dict(checkpoint["q_network"])
            self.target_network.load_state_dict(checkpoint["target_network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            self.last_episode = checkpoint.get("last_episode", 0)
            self.bucket_steps = checkpoint.get(
                "bucket_steps",
                [0] * self.bucket_count
            )
            self.balancing_steps = checkpoint.get("balancing_steps", 0)
            self.total_steps = checkpoint.get("total_steps", 0)
            self.env_steps = checkpoint.get("env_steps", 0)

            print("Loaded checkpoint:")
            print(f"  episode      : {self.last_episode}")
            print(f"  total_steps  : {self.total_steps:,}")
            print(f"  balancing_steps : {self.balancing_steps:,}")
            print(f"  bucket_steps : {self.bucket_steps}")
            print(f"  memory size : {[len(bucket) for bucket in self.memory]}")
            print(f"  losses       : {len(self.losses)} entries")

    def get_training_distribution(self) -> str:
        total = sum(self.updates_per_bucket)

        lines = [
            "\nTraining Distribution:",
            f"{'Bucket':<10}{'Focused':>12}"
            f"{'Balancing':>12}{'Total':>12}{'Share':>10}",
            "-" * 56,
        ]

        for i, count in enumerate(self.updates_per_bucket):
            share = 100 * count / total if total else 0

            lines.append(
                f"{i + 1:<10}"
                f"{self.bucket_steps[i]:>12,}"
                f"{self.balancing_updates_per_bucket[i]:>12,}"
                f"{count:>12,}"
                f"{share:>9.1f}%"
            )

        return "\n".join(lines) + "\n"

    def evaluate(
        self,
        env: GymnasiumWrapper,
        games_per_board: int = 20,
        force_board_size: int | None = None
    ) -> str:
        """
        Evaluate the greedy policy across sufficiently trained buckets.

        Metrics:
            - Average score
            - Games ending with length < 10 or >= 10
            - Deaths, training stagnation stops, evaluation caps
            - Average steps per game
        """
        min_bucket_steps = self.end_total_steps * 0.2
        all_results = []
        report = []

        if force_board_size is None:
            buckets_to_evaluate = [
                (i, board_sizes)
                for i, board_sizes in enumerate(BOARD_BUCKETS)
                if self.bucket_steps[i] >= min_bucket_steps
            ]

        else:
            bucket_index = next(
                (
                    i
                    for i, board_sizes in enumerate(BOARD_BUCKETS)
                    if force_board_size in board_sizes
                ),
                None,
            )

            buckets_to_evaluate = [
                (None, [force_board_size])
            ]

        header = (
            f"{'Board':<11}"
            f"{'Games':>7}"
            f"{'Avg Score':>12}"
            f"{'Best':>8}"
            f"{'Worst':>8}"
            f"{'Len <10':>10}"
            f"{'Len >=10':>11}"
            f"{'Death':>8}"
            f"{'Stopped':>10}"
            f"{'Capped':>9}"
            f"{'Avg Steps':>12}"
        )

        separator = "-" * len(header)

        def format_row(label: str, results: list[tuple]) -> str:
            """Format one board, bucket, or overall summary."""
            scores = [result[0] for result in results]
            lengths = [result[1] for result in results]
            endings = [result[2] for result in results]
            steps = [result[3] for result in results]

            return (
                f"{label:<11}"
                f"{len(results):>7}"
                f"{np.mean(scores):>12.2f}"
                f"{max(scores):>8}"
                f"{min(scores):>8}"
                f"{sum(length < 10 for length in lengths):>10}"
                f"{sum(length >= 10 for length in lengths):>11}"
                f"{endings.count('death'):>8}"
                f"{endings.count('stopped'):>10}"
                f"{endings.count('capped'):>9}"
                f"{np.mean(steps):>12.1f}"
            )

        for bucket_index, board_sizes in buckets_to_evaluate:
            bucket_results = []

            if bucket_index is None:
                report.append(
                    f"\nCustom Board | Size: "
                    f"{force_board_size}x{force_board_size}"
                )
            else:
                bucket_steps = self.bucket_steps[bucket_index]
                report.append(
                    f"\nBucket {bucket_index + 1} "
                    f"| Steps: {bucket_steps:,} "
                    f"| Epsilon: {self.get_epsilon(bucket_index):.4f}"
                )

            report.append(separator)
            report.append(header)
            report.append(separator)

            for board_size in board_sizes:
                results = []
                max_eval_steps = max(1_000, board_size * board_size)

                for _ in range(games_per_board):
                    env.reset(board_size=board_size)
                    state = env.board.get_snake_vision_vector()
                    max_snake_length = len(env.board.body)
                    ending = "capped"

                    step_count = 0
                    for step_count in range(1, max_eval_steps + 1):
                        action = Action(
                            self.get_action(
                                state,
                                explore_mode=False,
                            )
                        )

                        state, _, is_dead, stop_training = env.step(
                            action,
                            training=True,
                        )
                        max_snake_length = max(
                            max_snake_length,
                            len(env.board.body)
                        )

                        if is_dead:
                            ending = "death"
                            break

                        if stop_training:
                            ending = "stopped"
                            break

                    score, _, _, _ = env.board.get_board_statistics()
                    results.append((
                        score,
                        max_snake_length,
                        ending,
                        step_count,
                    ))

                bucket_results.extend(results)
                all_results.extend(results)
                report.append(
                    format_row(
                        f"{board_size}x{board_size}",
                        results,
                    )
                )

            report.append(separator)
            report.append(format_row("BUCKET AVG", bucket_results))
            report.append(separator)

        if not all_results:
            return "No buckets sufficiently trained for evaluation."

        report.append("")
        report.append("=" * len(header))
        report.append(header)
        report.append(separator)
        report.append(format_row("OVERALL", all_results))
        report.append("=" * len(header))
        report.append("")

        return "\n".join(report)

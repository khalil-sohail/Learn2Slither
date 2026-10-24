import os
from typing import Any


class Config:
    """
    A class to hold the configuration for the Snake game engine.
    it encapsulates all the parameters that can be configured for the game.
    """
    def __init__(self):
        # Environment Rules
        self.board_size = 10
        self.fps_cap = 15

        self.reward_green_apple = +5.00
        self.reward_red_apple = -1.00
        self.reward_death = -10.0
        self.reward_step = -0.02

        # DQN Hyperparameters
        self.learning_rate = 0.0001
        self.gamma = 0.995

        self.new_weights = "./models/"
        self.loading_wieghts = None
        self.save_weights = True

        # training parameters
        self.target_update_every = 1_000
        self.training_sessions = None
        self.end_total_steps = 2_000
        self.batch_size = 128
        self.train_every = 10

        # Application & File I/O State
        self.evaluate_model = 50
        self.mode = "LOBBY"

        # general configuration
        self.print_vision = False

        # expected input/types for each configuration parameter
        self._expected_types = {
            "board_size": int,
            "fps_cap": int,
            "reward_green_apple": float,
            "reward_red_apple": float,
            "reward_death": float,
            "reward_step": float,
            "learning_rate": float,
            "gamma": float,
            "loading_wieghts": type(None),
            "new_weights": type(None),
            "save_weights": bool,
            "end_total_steps": int,
            "batch_size": int,
            "train_every": int,
            "target_update_every": int,
            "training_sessions": int,
            "evaluate_model": int,
            "mode": str,
            "print_vision": bool
        }
        self._expected_input = {
            "board_size": {
                "min": 5,
                "max": 500
            },
            "fps_cap": {
                "min": 0,
                "max": 1000
            },
            "reward_green_apple": {
                "min": -10.0,
                "max": 10.0
            },
            "reward_red_apple": {
                "min": -10.0,
                "max": 10.0
            },
            "reward_death": {
                "min": -10.0,
                "max": 10.0
            },
            "reward_step": {
                "min": -1.0,
                "max": 1.0
            },
            "learning_rate": {
                "min": 0.0001,
                "max": 1.0
            },
            "gamma": {
                "min": 0.0,
                "max": 1.0
            },
            "loading_wieghts": {
                "allowed": [str, None, "must_exist"]
            },
            "new_weights": {
                "allowed": [str, None, "optional"]
            },
            "save_weights": {
                "allowed": [True, False]
            },
            "end_total_steps": {
                "min": 0,
                "max": 10_000_000
            },
            "batch_size": {
                "min": 1,
                "max": 10_000
            },
            "train_every": {
                "min": 1,
                "max": 10_000
            },
            "target_update_every": {
                "min": 1,
                "max": 10_000
            },
            "training_sessions": {
                "min": 1,
                "max": 1000_000
            },
            "evaluate_model": {
                "min": 1,
                "max": 1000
            },
            "mode": {
                "allowed": [
                    "LOBBY",
                    "CONFIG",
                    "EVALUATE",
                    "TRAIN",
                    "HUMAN_PLAYING",
                    "WATCH_TRAIN",
                    "AGENT_PLAYING_EX",
                    "AGENT_PLAYING_EI",
                    "EXIT"
                ]
            },
            "print_vision": {
                "allowed": [True, False]
            }
        }

    def update(self, key: str, value: Any) -> None:
        """
        Updates the configuration,
            validating against expected types and constraints.
        Raises a ValueError if conversion or boundary validation fails.
        """
        if key not in self._expected_types:
            raise ValueError(f"Invalid configuration key: {key}")

        expected_type = self._expected_types[key]
        validation_rules = self._expected_input[key]

        if str(value).lower() in ["none", ""]:
            if (
                "allowed" in validation_rules
                and None in validation_rules["allowed"]
            ):
                setattr(self, key, None)
                return
            elif expected_type is type(None):
                setattr(self, key, None)
                return

        try:
            if expected_type == bool:
                if isinstance(value, str):
                    if value.lower() in ["true", "1", "yes"]:
                        casted_value = True
                    elif value.lower() in ["false", "0", "no"]:
                        casted_value = False
                    else:
                        raise ValueError(
                            "Invalid boolean string for "
                            f"'{key}': {value}"
                        )
                else:
                    casted_value = bool(value)

            elif (
                expected_type is type(None)
                and "allowed" in validation_rules
                and str in validation_rules["allowed"]
            ):
                path = str(value)
                if (
                    not os.path.exists(path)
                    and "must_exist" in validation_rules["allowed"]
                ):
                    raise ValueError(f"File path does not exist: {path}")
                casted_value = path
            else:
                casted_value = expected_type(value)

        except (ValueError, TypeError) as e:
            raise ValueError(
                f"Invalid input type for '{key}'. "
                f"Expected {expected_type.__name__}. "
                f"Reason: {e}"
            )

        if "min" in validation_rules and "max" in validation_rules:
            if (
                not (validation_rules["min"]
                     <= casted_value <= validation_rules["max"])
            ):
                raise ValueError(
                    f"Value for '{key}' out of bounds. "
                    f"Must be between {validation_rules['min']} "
                    f"and {validation_rules['max']}."
                )

        if "allowed" in validation_rules:
            is_valid = False
            for allowed_item in validation_rules["allowed"]:
                if (
                    isinstance(allowed_item, type)
                    and isinstance(casted_value, allowed_item)
                ):
                    is_valid = True
                    break
                elif casted_value == allowed_item:
                    is_valid = True
                    break

            if not is_valid:
                raise ValueError(
                    f"Invalid value for '{key}'. "
                    f"Allowed options: {validation_rules['allowed']}"
                )

        setattr(self, key, casted_value)

    def get(self, key):
        """
        Retrieves the value for a given configuration key.
        :param key: The configuration key to retrieve.
        :return: The value associated with the key.
        """

        if hasattr(self, key):
            return getattr(self, key)
        else:
            raise ValueError(f"Invalid configuration key: {key}")

    def to_dict(self):
        """
        Returns the configuration as a dictionary.
        :return: A dictionary representation of the configuration.
        """

        return {
            "board_size": self.board_size,
            "fps_cap": self.fps_cap,
            "reward_green_apple": self.reward_green_apple,
            "reward_red_apple": self.reward_red_apple,
            "reward_death": self.reward_death,
            "reward_step": self.reward_step,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "loading_wieghts": self.loading_wieghts,
            "new_weights": self.new_weights,
            "save_weights": self.save_weights,
            "end_total_steps": self.end_total_steps,
            "batch_size": self.batch_size,
            "train_every": self.train_every,
            "target_update_every": self.target_update_every,
            "training_sessions": self.training_sessions,
            "evaluate_model": self.evaluate_model,
            "mode": self.mode,
            "print_vision": self.print_vision
        }

    def from_dict(self, config_dict):
        """
        Updates the configuration from a dictionary.
        :param config_dict:
            A dictionary containing configuration keys and values.
        """

        for key, value in config_dict.items():
            self.update(key, value)


GLOBAL_CONFIG = Config()

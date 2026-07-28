import os

class Config:
    """
    A class to hold the configuration for the Snake game engine.
    This class encapsulates all the parameters that can be configured for the game,
    including environment rules, neural network hyperparameters, and application state.
    """
    def __init__(self):
        # Environment Rules
        self.board_size = 10
        self.fps_cap = 15
        self.reward_green_apple = 1.0
        self.reward_red_apple = -0.9
        self.reward_death = -2.0
        self.reward_step = -0.01
        
        # Neural Network Hyperparameters
        self.episodes = 5000
        self.batch_size = 64
        self.learning_rate = 0.00005
        self.gamma = 0.95
        self.epsilon_decay = 0.9995
        self.memory_capacity = 100000
        
        # Application & File I/O State
        self.mode = "LOBBY"
        self.target_weights = None
        self.new_weights = None
        self.save_interval = 1000

        # general configuration
        self.print_vision = False
        # self.window_size = ()

        # expected input/types for each configuration parameter
        self._expected_types = {
            "board_size": int,
            "fps_cap": int,
            "reward_green_apple": float,
            "reward_red_apple": float,
            "reward_death": float,
            "reward_step": float,
            "episodes": int,
            "learning_rate": float,
            "gamma": float,
            "epsilon_decay": float,
            "memory_capacity": int,
            "mode": str,
            "target_weights": type(None),
            "new_weights": type(None),
            "save_interval": int,
            "print_vision": bool
        }
        self._expected_input = {
            "board_size": {
                "min": 5,
                "max": 100
            },
            "fps_cap": {
                "min": 1,
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
            "episodes": {
                "min": 1,
                "max": 10000
            },
            "learning_rate": {
                "min": 0.0001,
                "max": 1.0
            },
            "gamma": {
                "min": 0.0,
                "max": 1.0
            },
            "epsilon_decay": {
                "min": 0.0,
                "max": 1.0
            },
            "memory_capacity": {
                "min": 1000,
                "max": 1000000
            },
            "mode": {
                "allowed": [
                    "LOBBY",
                    "CONFIG",
                    "TRAIN",
                    "HUMAN_PLAYING",
                    "WATCH_TRAIN",
                    "AGENT_PLAYING",
                    "EXIT"
                ]
            },
            "target_weights": {
                "allowed": [str, None, "must_exist"]
            },
            "new_weights": {
                "allowed": [str, None, "optional"]
            },
            "save_interval": {
                "min": 1,
                "max": 10000
            },
            "print_vision": {
                "allowed": [True, False]
            }
        }

    def update(self, key, value):
        """
        Updates the configuration, validating against expected types and constraints.
        Raises a ValueError if conversion or boundary validation fails.
        """
        if key not in self._expected_types:
            raise ValueError(f"Invalid configuration key: {key}")

        expected_type = self._expected_types[key]
        validation_rules = self._expected_input[key]

        if str(value).lower() in ["none", ""]:
            if "allowed" in validation_rules and None in validation_rules["allowed"]:
                setattr(self, key, None)
                return
            elif expected_type == type(None):
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
                        raise ValueError(f"Invalid boolean string for '{key}': {value}")
                else:
                    casted_value = bool(value)
            
            elif expected_type == type(None) and "allowed" in validation_rules and str in validation_rules["allowed"]:
                path = str(value)
                if not os.path.exists(path) and "must_exist" in validation_rules["allowed"]:
                    raise ValueError(f"File path does not exist: {path}")
                casted_value = path
            else:
                casted_value = expected_type(value)
                
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid input type for '{key}'. Expected {expected_type.__name__}. Reason: {e}")

        if "min" in validation_rules and "max" in validation_rules:
            if not (validation_rules["min"] <= casted_value <= validation_rules["max"]):
                raise ValueError(
                    f"Value for '{key}' out of bounds. "
                    f"Must be between {validation_rules['min']} and {validation_rules['max']}."
                )

        if "allowed" in validation_rules:
            is_valid = False
            for allowed_item in validation_rules["allowed"]:
                if isinstance(allowed_item, type) and isinstance(casted_value, allowed_item):
                    is_valid = True
                    break
                elif casted_value == allowed_item:
                    is_valid = True
                    break
            
            if not is_valid:
                raise ValueError(f"Invalid value for '{key}'. Allowed options: {validation_rules['allowed']}")

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
            "epsilon_decay": self.epsilon_decay,
            "memory_capacity": self.memory_capacity,
            "mode": self.mode,
            "target_weights": self.target_weights,
            "save_interval": self.save_interval,
            "print_vision": self.print_vision
        }
    
    def from_dict(self, config_dict):
        """
        Updates the configuration from a dictionary.
        :param config_dict: A dictionary containing configuration keys and values.
        """
        
        for key, value in config_dict.items():
            self.update(key, value)

GLOBAL_CONFIG = Config()
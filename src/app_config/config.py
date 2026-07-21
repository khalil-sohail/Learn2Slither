
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
        self.reward_red_apple = -0.5
        self.reward_death = -2.0
        self.reward_step = -0.01
        
        # Neural Network Hyperparameters
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.epsilon_decay = 0.995
        self.memory_capacity = 100000
        
        # Application & File I/O State
        self.mode = "play"
        self.target_weights = None
        self.save_interval = 100
    
    def update(self, key, value):
        """
        Updates the configuration, automatically casting the input to the correct data type.
        Raises a ValueError if the conversion fails.
        """
        if not hasattr(self, key):
            raise ValueError(f"Invalid configuration key: {key}")

        current_value = getattr(self, key)

        if current_value is None:
            if value == "" or str(value).lower() == "none":
                setattr(self, key, None)
            else:
                setattr(self, key, str(value))
            return

        expected_type = type(current_value)

        try:
            casted_value = expected_type(value)
            setattr(self, key, casted_value)
        except ValueError:
            raise ValueError(f"Invalid input: Expected a {expected_type.__name__}")
    
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
            "save_interval": self.save_interval
        }
    
    def from_dict(self, config_dict):
        """
        Updates the configuration from a dictionary.
        :param config_dict: A dictionary containing configuration keys and values.
        """
        
        for key, value in config_dict.items():
            self.update(key, value)

GLOBAL_CONFIG = Config()
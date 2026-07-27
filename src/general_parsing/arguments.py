# this file is used to parse the program arguments and return the parsed arguments to the main function

# expected arguments
# """
#     mode -m: mode of operation (e.g. "train", "play", "watch-train", "watch-exploration-train", "watch-exploitation").
#     fps -f: frames per second (e.g. 30, 60). only used in "play" and "watch-train" modes
#     board -b: board size (e.g. 10, 50, 100). represents the number of cells in the board.
#               the board is a square, so the total number of cells is board^2.
#     -h: help
# """

from app_config import GLOBAL_CONFIG

def checkMode(mode):
    """
    Validates the provided mode against the allowed modes in GLOBAL_CONFIG.
    :param mode: The mode to validate.
    :return: The validated mode if it's allowed.
    :raises ValueError: If the mode is not allowed.
    """
    valid_modes = ["LOBBY", "CONFIG", "TRAIN", "HUMAN_PLAYING", "WATCH_TRAIN", "AGENT_PLAYING", "EXIT"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Valid modes are: {valid_modes}")
    return mode

def parseArguments(argv) -> bool:
    """
    Parse command-line arguments and mutate the engine_config dictionary.
    :param argv: List of command-line arguments (excluding the script name).
    """
    arg_map: dict[str, tuple] = {
        "-m": (
            ("mode", checkMode),
            "Expected Modes: LOBBY, CONFIG, TRAIN, HUMAN_PLAYING, WATCH_TRAIN, AGENT_PLAYING, EXIT"
        ),
        "--mode": (
            ("mode", checkMode),
            "Expected Modes: LOBBY, CONFIG, TRAIN, HUMAN_PLAYING, WATCH_TRAIN, AGENT_PLAYING, EXIT"
        ),
        "-b": (
            ("board_size", int),
            "Expected Board Size: integer between 5 and 100"
        ),
        "--board": (
            ("board_size", int),
            "Expected Board Size: integer between 5 and 100"
        ),
        "-f": (
            ("fps_cap", int),
            "Expected FPS: integer"
        ),
        "--fps": (
            ("fps_cap", int),
            "Expected FPS: integer"
        ),
        
        # Neural Network Flags
        "-e": (
            ("episodes", int),
            "Expected Episodes: integer"
        ),
        "--batch-size": (
            ("batch_size", int),
            "Expected Batch Size: integer"
        ),
        "--lr": (
            ("learning_rate", float),
            "Expected Learning Rate: float"
        ),
        "--gamma": (
            ("gamma", float),
            "Expected Gamma: float"
        ),
        "--epsilon": (
            ("epsilon_decay", float),
            "Expected Epsilon Decay: float"
        ),
        "--memory": (
            ("memory_capacity", int),
            "Expected Memory Capacity: integer"
        ),
        
        # Reward Flags
        "--reward-apple": (
            ("reward_apple", float),
            "Expected Reward for Eating Apple: float"
        ),
        "--reward-death": (
            ("reward_death", float),
            "Expected Reward for Dying: float"
        ),
        "--reward-red-apple": (
            ("reward_red_apple", float),
            "Expected Reward for Eating Red Apple: float"
        ),
        "--reward-green-apple": (
            ("reward_green_apple", float),
            "Expected Reward for Eating Green Apple: float"
        ),
        "--reward-step": (
            ("reward_step", float),
            "Expected Reward for Each Step: float"
        ),
        
        # File I/O Flags
        "--weights": (
            ("target_weights", str),
            "Expected Weights File: string"
        ),
        "--new-weights": (
            ("new_weights", str),
            "Expected New Weights File: string"
        ),
        "--save-interval": (
            ("save_interval", int),
            "Expected Save Interval: integer"
        ),

        # General Configuration
        "--print-vision": (
            ("print_vision", str), "Expected vision printing option: string (True/False)"
        )
    }

    i = 0
    while i < len(argv):
        arg = argv[i]
        
        if arg in ("-h", "--help"):
            print("Help requested. Displaying help information...")
            print("Usage: python main.py [options]")
            print("Environment Options:")
            print("  -m, --mode        Mode ('LOBBY', 'CONFIG', 'TRAIN', 'HUMAN_PLAYING', 'WATCH_TRAIN', 'AGENT_PLAYING', 'EXIT')")
            print("  -b, --board       Board size (e.g., 10, 20)")
            print("  -f, --fps         Frames per second limit")
            print("AI Hyperparameters:")
            print("  --lr              Learning rate (default: 0.001)")
            print("  --gamma           Discount factor (default: 0.99)")
            print("  --epsilon         Epsilon decay rate (default: 0.995)")
            print("  --memory          Replay buffer capacity")
            print("Rewards:")
            print("  --reward-red-apple    Float reward for eating red apple (default: 1.0)")
            print("  --reward-green-apple    Float reward for eating green apple (default: 1.0)")
            print("  --reward-death    Float penalty for dying (default: -1.0)")
            print("  --reward-step     Float penalty per tick (default: -0.01)")
            print("Files:")
            print("  --weights         Path to .pt file to load")
            print("  --new-weights     Path to .pt file to load")
            print("  --save-interval   Episodes between saving weights")
            print("General Configuration:")
            print("  --print-vision    Whether to print vision information (default: False)")
            return False

        elif arg in arg_map:
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value for argument: {arg}")
            
            (dict_key, cast_func), explanation = arg_map[arg]
            try:
                GLOBAL_CONFIG.update(dict_key, cast_func(argv[i + 1]))
            except ValueError:
                raise ValueError(f"Invalid type for {arg}. {explanation}.")
                
            i += 2
            continue
            
        else:
            raise ValueError(f"Unknown argument: {arg}. Ignoring.")
    
    return True

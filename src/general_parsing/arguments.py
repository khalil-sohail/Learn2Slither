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
    """modes: LOBBY, CONFIG, TRAIN, HUMAN_PLAYING, WATCH_TRAIN, AGENT_PLAYING, EXIT"""
    valid_modes = ["LOBBY", "CONFIG", "TRAIN", "HUMAN_PLAYING", "WATCH_TRAIN", "AGENT_PLAYING", "EXIT"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Valid modes are: {valid_modes}")
    return mode

def parseArguments(argv):
    """
    Parse command-line arguments and mutate the engine_config dictionary.
    :param argv: List of command-line arguments (excluding the script name).
    """
    arg_map = {
        "-m": ("mode", checkMode),
        "--mode": ("mode", checkMode),
        "-b": ("board_size", int),
        "--board": ("board_size", int),
        "-f": ("fps_cap", int),
        "--fps": ("fps_cap", int),
        
        # Neural Network Flags
        "-e": ("epoch", int),
        "--lr": ("learning_rate", float),
        "--gamma": ("gamma", float),
        "--epsilon": ("epsilon_decay", float),
        "--memory": ("memory_capacity", int),
        
        # Reward Flags
        "--reward-apple": ("reward_apple", float),
        "--reward-death": ("reward_death", float),
        "--reward-red-apple": ("reward_red_apple", float),
        "--reward-green-apple": ("reward_green_apple", float),
        "--reward-step": ("reward_step", float),
        
        # File I/O Flags
        "--weights": ("target_weights", str),
        "--save-interval": ("save_interval", int),

        # General Configuration
        "--print-vision": ("print_vision", str)
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
            print("  --save-interval   Episodes between saving weights")
            print("General Configuration:")
            print("  --print-vision    Whether to print vision information (default: False)")
            return None

        elif arg in arg_map:
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value for argument: {arg}")
            
            dict_key, cast_func = arg_map[arg]
            
            try:
                GLOBAL_CONFIG.update(dict_key, cast_func(argv[i + 1]))
            except ValueError:
                raise ValueError(f"Invalid type for {arg}. Expected {cast_func.__doc__}.")
                
            i += 2
            continue
            
        else:
            raise ValueError(f"Unknown argument: {arg}. Ignoring.")

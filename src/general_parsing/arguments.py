from app_config import GLOBAL_CONFIG
import os


def checkMode(mode):
    """
    Validates the provided mode against the allowed modes in GLOBAL_CONFIG.
    :param mode: The mode to validate.
    :return: The validated mode if it's allowed.
    :raises ValueError: If the mode is not allowed.
    """
    valid_modes = [
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

    if mode not in valid_modes:
        raise ValueError(
            f"Invalid mode: {mode}. "
            f"Valid modes are: {valid_modes}"
        )
    return mode


def parseBool(value):
    """
    Converts a string representation of a boolean to an actual boolean.
    :param value: The string to convert.
    :return: True or False based on the input string.
    :raises ValueError: If the string does not represent a boolean.
    """
    if isinstance(value, bool):
        return value
    if value.lower() in ["true", "1", "yes"]:
        return True
    elif value.lower() in ["false", "0", "no"]:
        return False
    else:
        raise ValueError(
            f"Invalid boolean value: {value}. "
            f"Expecting 'true', 'false', '1', '0', 'yes', or 'no'."
        )


def checkFilePath(path):
    """
    Validates that the provided path exists and is a file.
    :param path: The file path to validate.
    :return: The validated file path.
    :raises ValueError: If the path does not exist or is not a file.
    """
    if not os.path.exists(path):
        raise ValueError(f"File path does not exist: {path}")
    if not os.path.isfile(path):
        raise ValueError(f"Path is not a file: {path}")
    return path


def checkDirectoryPath(path):
    """
    Validates that the provided path exists and is a directory.
    :param path: The directory path to validate.
    :return: The validated directory path.
    :raises ValueError: If the path does not exist or is not a directory.
    """
    if not os.path.exists(path):
        raise ValueError(f"Directory path does not exist: {path}")
    if not os.path.isdir(path):
        raise ValueError(f"Path is not a directory: {path}")
    return path


def parseArguments(argv) -> bool:
    """
    Parse command-line arguments and mutate the engine_config dictionary.
    :param argv: List of command-line arguments (excluding the script name).
    """
    arg_map: dict[str, tuple] = {
        "-m": (
            ("mode", checkMode),
            "Expecting Modes: LOBBY, CONFIG, EVALUATE, "
            "TRAIN, HUMAN_PLAYING, WATCH_TRAIN, "
            "AGENT_PLAYING_EX, AGENT_PLAYING_EI, EXIT"
        ),
        "--mode": (
            ("mode", checkMode),
            "Expecting Modes: LOBBY, CONFIG, EVALUATE, "
            "TRAIN, HUMAN_PLAYING, WATCH_TRAIN, "
            "AGENT_PLAYING_EX, AGENT_PLAYING_EI, EXIT"
        ),
        "-eval": (
            ("evaluate_model", int),
            "Expecting games per board: integer"
        ),
        "-b": (
            ("board_size", int),
            "Expecting Board Size: integer between 5 and 100"
        ),
        "--board": (
            ("board_size", int),
            "Expecting Board Size: integer between 5 and 100"
        ),
        "-f": (
            ("fps_cap", int),
            "Expecting FPS: integer"
        ),
        "--fps": (
            ("fps_cap", int),
            "Expecting FPS: integer"
        ),

        # Neural Network Flags
        "--lr": (
            ("learning_rate", float),
            "Expecting Learning Rate: float"
        ),
        "--gamma": (
            ("gamma", float),
            "Expecting Gamma: float"
        ),
        "--total-steps": (
            ("end_total_steps", int),
            "Expecting Total optimizer updates for "
            "each pocket at the end of training: integer"
        ),
        "--batch-size": (
            ("batch_size", int),
            "Expecting Batch Size: integer"
        ),
        "--train-every": (
            ("train_every", int),
            "Expecting Training Frequency: integer"
        ),
        "--target-update-every": (
            ("target_update_every", int),
            "Expecting Target Network Update Frequency: integer"
        ),
        "--training-sessions": (
            ("training_sessions", int),
            "Expecting Training Sessions: integer"
        ),

        # Reward Flags
        "--reward-apple": (
            ("reward_apple", float),
            "Expecting Reward for Eating Apple: float"
        ),
        "--reward-death": (
            ("reward_death", float),
            "Expecting Reward for Dying: float"
        ),
        "--reward-red-apple": (
            ("reward_red_apple", float),
            "Expecting Reward for Eating Red Apple: float"
        ),
        "--reward-green-apple": (
            ("reward_green_apple", float),
            "Expecting Reward for Eating Green Apple: float"
        ),
        "--reward-step": (
            ("reward_step", float),
            "Expecting Reward for Each Step: float"
        ),

        # File I/O Flags
        "--weights": (
            ("loading_wieghts", checkFilePath),
            "Expecting a string representing a file path to "
            "load weights (e.g., './models/snake_model.pt')"
        ),
        "--new-weights": (
            ("new_weights", checkDirectoryPath),
            "Expecting a string representing a directory path "
            "to save new weights (e.g., './models/')"
        ),
        "--save-weights": (
            ("save_weights", parseBool),
            "Expecting a boolean value (True/False) to determine "
            "whether to save weights"
        ),

        # General Configuration
        "--print-vision": (
            ("print_vision", parseBool), "Expecting a "
            "boolean value (True/False) to determine "
            "whether to print vision information"
        )
    }

    i = 0
    while i < len(argv):
        arg = argv[i]

        if arg in ("-h", "--help"):
            print("Help requested. Displaying help information...")
            print("Usage: python main.py [options]")
            print("Environment Options:")
            print(
                "  -m, --mode        Mode ('LOBBY', 'CONFIG',"
                " 'EVALUATE', 'TRAIN', 'HUMAN_PLAYING', 'WATCH_TRAIN',"
                " 'AGENT_PLAYING_EX', 'AGENT_PLAYING_EI', 'EXIT')"
            )
            print(
                "  -eval             "
                "Number of games per board for evaluation (default: 50)"
            )
            print("  -b, --board       Board size (e.g., 10, 20)")
            print("  -f, --fps         Frames per second limit")
            print("AI Hyperparameters:")
            print("  --lr              Learning rate (default: 0.001)")
            print("  --gamma           Discount factor (default: 0.995)")
            print(
                "  --total-steps     "
                "Total training steps for each bucket (default: 100,000)"
            )
            print("  --batch-size      Batch size for training (default: 64)")
            print(
                "  --train-every     "
                "Steps between training updates (default: 5)"
            )
            print(
                "  --target-update-every Steps between target network updates "
                "(default: 1,000)"
            )
            print(
                "  --training-sessions number of training "
                "sessions for the 10x10 board (default: None, meaning "
                "steps-curriculum training will be used)"
            )
            print("Rewards:")
            print(
                "  --reward-red-apple    "
                "Float reward for eating red apple (default: -1.0)"
            )
            print(
                "  --reward-green-apple    "
                "Float reward for eating green apple (default: +5.0)"
            )
            print(
                "  --reward-death    "
                "Float penalty for dying (default: -1.0)"
            )
            print(
                "  --reward-step     "
                "Float penalty per tick (default: -0.01)"
            )
            print("Files:")
            print("  --weights         Path to .pt file to load")
            print(
                "  --new-weights     "
                "Path to save new weights (default: './models/')"
            )
            print(
                "  --save-weights    "
                "Whether to save weights (default: True)"
            )
            print("General Configuration:")
            print(
                "  --print-vision    "
                "Whether to print vision information (default: False)"
            )
            return False

        elif arg in arg_map:
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value for argument: {arg}")

            (dict_key, cast_func), explanation = arg_map[arg]
            try:
                casted_value = cast_func(argv[i + 1])
            except ValueError:
                raise ValueError(
                    f"`{type(argv[i + 1]).__name__}` is "
                    f"Invalid type for {arg}. {explanation}."
                )
            try:
                GLOBAL_CONFIG.update(dict_key, casted_value)
            except ValueError as e:
                raise ValueError(e)

            i += 2
            continue

        else:
            raise ValueError(f"Unknown argument: {arg}. Ignoring.")

    return True

# this file is used to parse the program arguments and return the parsed arguments to the main function

# expected arguments
# """
#     mode -m: mode of operation (e.g. "train", "play", "watch-train")
#     fps -f: frames per second (e.g. 30, 60). only used in "play" and "watch-train" modes
#     board -b: board size (e.g. 10, 50, 100). represents the number of cells in the board.
#               the board is a square, so the total number of cells is board^2.
#     -h: help
# """


def checkMode(mode):
    """
    Check if the provided mode is valid.
    """
    valid_modes = ["train", "play", "watch-train"]
    if mode not in valid_modes:
        raise ValueError(f"Invalid mode: {mode}. Valid modes are: {valid_modes}")
    return mode

def parseArguments(argv):
    """
    Parse command-line arguments and return a dictionary of parsed values.
    """

    args = {
        "mode": "play",
        "fps": 30,
        "board": 10
    }
    i = 0

    while i < len(argv):
        arg = argv[i]
        if arg in ("-h", "--help"):
            print("Help requested. Displaying help information...")
            print("Usage: python main.py -m <mode> -f <fps> -b <board>")
            print("Options:")
            print("  -m, --mode    Mode of operation (e.g., 'train', 'play', 'watch-train')")
            print("  -f, --fps     Frames per second (e.g., 30, 60). Only used in 'play' and 'watch-train' modes")
            print("  -b, --board   Board size (e.g., 10, 50, 100). Represents the number of cells in the board.")
            print("                The board is a square, so the total number of cells is board^2.")
            print("  -h, --help    Show this help message and exit")
            return None
        elif arg in ("-m", "--mode"):
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value for argument: {arg}")
            args["mode"] = checkMode(argv[i + 1])
            i += 2
            continue
        elif arg in ("-f", "--fps"):
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value for argument: {arg}")
            args["fps"] = int(argv[i + 1])
            i += 2
            continue
        elif arg in ("-b", "--board"):
            if i + 1 >= len(argv):
                raise ValueError(f"Missing value for argument: {arg}")
            args["board"] = int(argv[i + 1])
            i += 2
            continue
        else:
            raise ValueError(f"Unknown argument: {arg}. Ignoring.")

    return args



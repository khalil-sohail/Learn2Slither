import sys
from generalParsing import parseArguments

def main(argv):
    try:
        args = parseArguments(argv)
        print(f"Mode: {args['mode']}, FPS: {args['fps']}, Board Size: {args['board']}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # game loop
        while True:
            if args['mode'] == "train":
                # env = GymnasiumWrapper(board_size=args['board'], render_mode=None)
                print("Training... To Do.")
            elif args['mode'] == "play":
                # env = GymnasiumWrapper(board_size=args['board'], render_mode='human')
                print("Playing... To Do.")
            elif args['mode'] == "watch-train":
                # env = GymnasiumWrapper(board_size=args['board'], render_mode='human')
                print("Watching training... To Do.")
            break  # Placeholder for the actual game loop logic
    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nGame interrupted by user. Exiting...")


if __name__ == "__main__":
    main(sys.argv[1:])

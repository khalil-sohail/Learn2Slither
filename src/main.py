from gymnasium_wrapper import GymnasiumWrapper
from graphics_engine import GraphicsEngine
from general_parsing import parseArguments
# from dqn_agent import DQNAgent
import pygame
import sys

from constants import Action

def get_human_input(current_action):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                return Action.UP
            elif event.key == pygame.K_DOWN:
                return Action.DOWN
            elif event.key == pygame.K_LEFT:
                return Action.LEFT
            elif event.key == pygame.K_RIGHT:
                return Action.RIGHT
                
    return current_action

def main(argv):
    try:
        args = parseArguments(argv)
        print(f"Mode: {args['mode']}, FPS: {args['fps']}, Board Size: {args['board']}")
    except ValueError as e:
        print(f"Error: {e}")

    if args['mode'] in ["play", "watch-train"]:
        gfx = GraphicsEngine(window_size=800, board_size=args['board'], fps=args['fps'])
    env = GymnasiumWrapper(board_size=args['board'])
    # agent = None
    
    # if args['mode'] in ["train", "watch-train"]:
    #     agent = DQNAgent()

    state, _ = env.reset()
    action = 3

    try:
        while True:
            if args['mode'] == "play":
                action = get_human_input(action) 
            # else:
            #     action = agent.select_action(state) 
            
            next_state, reward, done, truncated, info = env.step(action)
            print(f"Action: {action}, Reward: {reward}, Done: {done}, Truncated: {truncated}, Info: {info}")
            if args['mode'] == "train":
                pass 

            state = next_state

            if args['mode'] == "play" and done:
                gfx.game_over()
                break

            if args['mode'] in ["play", "watch-train"]:
                gfx.render_frame(state)

    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nGame interrupted by user. Exiting...")


if __name__ == "__main__":
    main(sys.argv[1:])

from gymnasium_wrapper import GymnasiumWrapper
from graphics_engine import GraphicsEngine
from general_parsing import parseArguments
from constants import Action, AppState
from app_config import GLOBAL_CONFIG
from dqn_agent import DQNAgent
import pygame
import sys

def get_input(current_action):
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.quit()
            sys.exit()
        elif GLOBAL_CONFIG.get('mode') == "play" and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                return Action.UP
            elif event.key == pygame.K_DOWN:
                return Action.DOWN
            elif event.key == pygame.K_LEFT:
                return Action.LEFT
            elif event.key == pygame.K_RIGHT:
                return Action.RIGHT
                
    return current_action

def game_loop(gfx: GraphicsEngine, env: GymnasiumWrapper, agent: DQNAgent, current_state: AppState):
    state, _ = env.reset()
    action = Action.RIGHT

    while True:
        action = get_input(action) 
        if current_state in [AppState.WATCH_TRAIN, AppState.WATCH_EXPLORATION_TRAIN, AppState.WATCH_EXPLOITATION]:
            action = agent.select_action(state)
        
        next_state, reward, is_dead, truncated, info, tail_direction = env.step(action)
        # print(f"Action: {action}, Reward: {reward}, is_dead: {is_dead}, Truncated: {truncated}, Info: {info}")
        if GLOBAL_CONFIG.get('mode') == "train":
            pass

        state = next_state

        if GLOBAL_CONFIG.get('mode') == "play" and is_dead:
            gfx.game_over()
            break

        if current_state in [AppState.PLAYING, AppState.WATCH_TRAIN, AppState.WATCH_EXPLORATION_TRAIN, AppState.WATCH_EXPLOITATION] and not is_dead:
            gfx.render_user_game_frame(state, action, tail_direction)
        elif current_state in [AppState.TRAIN]:
            gfx.render_agent_train()
    
    return AppState.LOBBY

def main(argv):
    try:
        parseArguments(argv)
    except ValueError as e:
        print(f"Error: {e}")

    current_state = AppState.LOBBY
    try:
        if GLOBAL_CONFIG.get('mode') in ["play", "watch-train", "watch-exploration-train", "watch-exploitation"]:
            gfx = GraphicsEngine(window_size=900)
        env = GymnasiumWrapper()
        agent = DQNAgent()

        while current_state != AppState.EXIT:
            if current_state == AppState.LOBBY:
                current_state = gfx.render_lobby()
            if current_state == AppState.CONFIG:
                current_state = gfx.render_config()
                env.reset()
                gfx.refresh_textures()
            if current_state in [AppState.PLAYING, AppState.WATCH_TRAIN, AppState.WATCH_EXPLORATION_TRAIN, AppState.WATCH_EXPLOITATION]:
                current_state = game_loop(gfx, env, agent, current_state)

    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nGame interrupted by user. Exiting...")


if __name__ == "__main__":
    main(sys.argv[1:])

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
            return current_action

        if GLOBAL_CONFIG.get('mode') == "play" and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP or event.key == pygame.K_w:
                return Action.UP
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                return Action.DOWN
            elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                return Action.LEFT
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                return Action.RIGHT

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                fps = GLOBAL_CONFIG.get('fps_cap') + 5
                fps = min(fps, 1000)
                GLOBAL_CONFIG.update('fps_cap', fps)
            elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                fps = GLOBAL_CONFIG.get('fps_cap') - 5
                fps = max(fps, 1)
                GLOBAL_CONFIG.update('fps_cap', fps)

    return current_action

def game_loop(gfx: GraphicsEngine, env: GymnasiumWrapper, agent: DQNAgent, current_state: AppState):
    state, _ = env.reset()
    action = Action.RIGHT

    while current_state in [AppState.PLAYING, AppState.WATCH_TRAIN, AppState.WATCH_EXPLORATION_TRAIN, AppState.WATCH_EXPLOITATION]:
        action = get_input(action) 
        if current_state in [AppState.WATCH_TRAIN, AppState.WATCH_EXPLORATION_TRAIN, AppState.WATCH_EXPLOITATION]:
            snake_vision = env.board.get_snake_vision_vector()
            action = agent.get_action(snake_vision)
            action = agent.get_test_action(state)
            if GLOBAL_CONFIG.get('print_vision'):
                if action is Action.UP:
                    print("UP")
                elif action is Action.DOWN:
                    print("DOWN")
                elif action is Action.LEFT:
                    print("LEFT")
                elif action is Action.RIGHT:
                    print("\nRIGHT\n")

        next_state, reward, is_dead, tail_direction = env.step(action)
        # print(f"Action: {action}, Reward: {reward}, is_dead: {is_dead}.")
        if GLOBAL_CONFIG.get('mode') == "train":
            pass

        state = next_state

        if current_state in [AppState.PLAYING, AppState.WATCH_TRAIN, AppState.WATCH_EXPLORATION_TRAIN, AppState.WATCH_EXPLOITATION]:
            board_score, green_apple_eaten, red_apple_eaten = env.board.get_board_statistics()
            current_state = gfx.render_user_game_frame(
                state,
                is_dead,
                current_state,
                board_score,
                green_apple_eaten,
                red_apple_eaten
            )
        elif current_state in [AppState.TRAIN]:
            current_state = gfx.render_agent_train()
    
    return AppState.LOBBY

def main(argv):
    try:
        parseArguments(argv)
        print("Configuration loaded successfully.")
        print(GLOBAL_CONFIG.to_dict())
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

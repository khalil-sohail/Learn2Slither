from collections import deque

from gymnasium_wrapper import GymnasiumWrapper
from graphics_engine import GraphicsEngine
from general_parsing import parseArguments
from constants import Action, AppState
from app_config import GLOBAL_CONFIG
from dqn_agent import DQNAgent
import pygame
import sys

def get_input(current_action, APP_STATE=AppState.HUMAN_PLAYING):
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            return current_action, AppState.EXIT

        if APP_STATE == AppState.HUMAN_PLAYING and event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP or event.key == pygame.K_w:
                current_action = Action.UP
            elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                current_action = Action.DOWN
            elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                current_action = Action.LEFT
            elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                current_action = Action.RIGHT

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                fps = GLOBAL_CONFIG.get('fps_cap') + 5
                fps = min(fps, 1000)
                GLOBAL_CONFIG.update('fps_cap', fps)
            elif event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                fps = GLOBAL_CONFIG.get('fps_cap') - 5
                fps = max(fps, 1)
                GLOBAL_CONFIG.update('fps_cap', fps)

    return current_action, APP_STATE

def game_loop(gfx: GraphicsEngine, env: GymnasiumWrapper, agent: DQNAgent, current_state: AppState):
    snake_vision = env.reset()
    action = Action.RIGHT

    while current_state in [AppState.HUMAN_PLAYING, AppState.WATCH_TRAIN, AppState.AGENT_PLAYING]:
        action, current_state = get_input(action, APP_STATE=current_state)
        if current_state == AppState.EXIT:
            break

        if current_state in [AppState.WATCH_TRAIN, AppState.AGENT_PLAYING]:
            action = agent.get_action(snake_vision)
            if GLOBAL_CONFIG.get('print_vision'):
                if action is Action.UP:
                    print("UP")
                elif action is Action.DOWN:
                    print("DOWN")
                elif action is Action.LEFT:
                    print("LEFT")
                elif action is Action.RIGHT:
                    print("RIGHT")

        next_snake_vision, reward, is_dead = env.step(action)
        if current_state == AppState.WATCH_TRAIN:
            agent.remember(snake_vision, action, next_snake_vision, reward, is_dead)
            agent.train_step(batch_size=32)

        if current_state in [AppState.HUMAN_PLAYING, AppState.WATCH_TRAIN, AppState.AGENT_PLAYING]:
            board_score, green_apple_eaten, red_apple_eaten, board = env.board.get_board_statistics()
            current_state = gfx.render_user_game_frame(
                board,
                is_dead,
                current_state,
                board_score,
                green_apple_eaten,
                red_apple_eaten
            )

        snake_vision = env.board.get_snake_vision_vector()

    return AppState.LOBBY

def train_game_loop(env: GymnasiumWrapper, agent: DQNAgent):
    snake_vision = env.reset()
    action = Action.RIGHT
    epoch = GLOBAL_CONFIG.get('epoch')
    scores = deque(maxlen=100)
    i = 0
    j = 0

    while i < epoch:
        action = agent.get_action(snake_vision)
        if GLOBAL_CONFIG.get('print_vision'):
            if action is Action.UP:
                print("UP")
            elif action is Action.DOWN:
                print("DOWN")
            elif action is Action.LEFT:
                print("LEFT")
            elif action is Action.RIGHT:
                print("RIGHT")

        next_snake_vision, reward, is_dead = env.step(action)
        agent.remember(snake_vision, action, next_snake_vision, reward, is_dead)
        if j % 5 == 0:
            agent.train_step(batch_size=32)


        # print(
        #     episode,
        #     agent.epsilon,
        #     len(agent.memory)
        # )
        if is_dead:
            i += 1
            board_score, green_apple_eaten, red_apple_eaten, _ = env.board.get_board_statistics()
            scores.append(board_score)
            if i % 100 == 0:
                avg_score = sum(scores) / len(scores)
                print(f"Epoch: {i}, Average Score (last 100): {avg_score:.2f}")
                scores.clear()
            snake_vision = env.reset()
        else:
            j += 1
            snake_vision = env.board.get_snake_vision_vector()

    return AppState.LOBBY

def main(argv):
    try:
        parseArguments(argv)
        print(GLOBAL_CONFIG.to_dict())
    except ValueError as e:
        print(f"Error: {e}")
        return

    window_size = 800
    current_state = AppState[GLOBAL_CONFIG.get('mode')]
    try:
        gfx = None
        if current_state != AppState.TRAIN:
            gfx = GraphicsEngine(window_size)
        env = GymnasiumWrapper()
        agent = DQNAgent()

        while current_state != AppState.EXIT:
            if current_state == AppState.TRAIN:
                current_state = train_game_loop(env, agent)
                if gfx is None:
                    gfx = GraphicsEngine(window_size)
            else:
                if current_state == AppState.LOBBY:
                    current_state = gfx.render_lobby()
                elif current_state == AppState.CONFIG:
                    current_state = gfx.render_config()
                    if current_state != AppState.CONFIG:
                        gfx.refresh_textures()
                elif current_state in [AppState.HUMAN_PLAYING, AppState.WATCH_TRAIN, AppState.AGENT_PLAYING]:
                    current_state = game_loop(gfx, env, agent, current_state)

    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nGame interrupted by user. Exiting...")


if __name__ == "__main__":
    main(sys.argv[1:])

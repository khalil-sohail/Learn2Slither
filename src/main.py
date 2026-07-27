from collections import deque

from gymnasium_wrapper import GymnasiumWrapper
from graphics_engine import GraphicsEngine
from general_parsing import parseArguments
from constants import Action, AppState
from app_config import GLOBAL_CONFIG
from dqn_agent import DQNAgent
import pygame
import sys
import numpy as np

def get_input(current_action: Action, APP_STATE: AppState) -> tuple[Action, AppState]:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            return current_action, AppState.EXIT

        if APP_STATE == AppState.HUMAN_PLAYING and event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_UP, pygame.K_w):
                current_action = Action.UP
            elif event.key in (pygame.K_DOWN, pygame.K_s):
                current_action = Action.DOWN
            elif event.key in (pygame.K_LEFT, pygame.K_a):
                current_action = Action.LEFT
            elif event.key in (pygame.K_RIGHT, pygame.K_d):
                current_action = Action.RIGHT

        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_p):
                fps = min(GLOBAL_CONFIG.get('fps_cap') + 5, 1000)
                GLOBAL_CONFIG.update('fps_cap', fps)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_m):
                fps = max(GLOBAL_CONFIG.get('fps_cap') - 5, 1000)
                GLOBAL_CONFIG.update('fps_cap', fps)

    return current_action, APP_STATE

def game_loop(gfx: GraphicsEngine, env: GymnasiumWrapper, agent: DQNAgent, current_state: AppState):
    snake_vision = env.reset()
    action: Action = Action.RIGHT

    while current_state in [
        AppState.HUMAN_PLAYING,
        AppState.WATCH_TRAIN,
        AppState.AGENT_PLAYING_EX,
        AppState.AGENT_PLAYING_EI
    ]:
        action, current_state = get_input(current_action=action, APP_STATE=current_state)
        if current_state == AppState.EXIT:
            break

        if current_state == AppState.WATCH_TRAIN:
            action = agent.get_action(snake_vision)
        elif current_state in [
            AppState.AGENT_PLAYING_EX,
            AppState.AGENT_PLAYING_EI
        ]:
            action = agent.get_action(snake_vision, explore_mode=(current_state == AppState.AGENT_PLAYING_EX))

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
            agent.train_step(batch_size=GLOBAL_CONFIG.get('batch_size'), update_epsilon=False)

        if current_state in [
            AppState.HUMAN_PLAYING,
            AppState.WATCH_TRAIN,
            AppState.AGENT_PLAYING_EX,
            AppState.AGENT_PLAYING_EI
        ]:
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
    episodes = GLOBAL_CONFIG.get('episodes')
    scores_length = 1000
    print_every = scores_length // 5
    scores = deque(maxlen=print_every)
    episode = 0
    global_step = 0
    update_epsilon = False
    debug: str = ""

    while episode < episodes:
        global_step += 1

        raw_action = agent.get_action(snake_vision)
        action = Action(raw_action)

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
        agent.train_step(batch_size=GLOBAL_CONFIG.get('batch_size'), update_epsilon=update_epsilon)
        update_epsilon = False

        if global_step % 10000 == 0:
            debug += "episode {}: Step: {}, Epsilon: {:.4f}, Memory Size: {}, Loss: {}\n".format(
                episode,
                global_step,
                agent.epsilon,
                len(agent.memory),
                agent.losses[-1] if agent.losses else None,
            )

        if is_dead:
            update_epsilon = True
            episode += 1
            board_score, green_apple_eaten, red_apple_eaten, _ = env.board.get_board_statistics()
            scores.append((board_score, green_apple_eaten, red_apple_eaten))

            if episode % print_every == 0 and scores:
                avg_score = sum(s[0] for s in scores) / len(scores)
                avg_green = sum(s[1] for s in scores) / len(scores)
                avg_red = sum(s[2] for s in scores) / len(scores)
                debug += "Episode: {}, Avg Score (last {}): {:.2f}, ".format(
                    episode, len(scores), avg_score
                )
                debug += "Avg Green Apples: {:.2f}, Avg Red Apples: {:.2f}\n".format(
                    avg_green, avg_red
                )

            snake_vision = env.reset()
        else:
            snake_vision = next_snake_vision
            

    if agent.losses:
        diffs = np.diff(agent.losses)
        debug += "Training completed after {} episodes. Loss diff mean: {:.6f}, std: {:.6f}\n".format(
            episodes, diffs.mean(), diffs.std()
        )
    else:
        debug += f"Training completed after {episodes} episodes. No losses recorded.\n"

    print(debug, end="")

    return AppState.LOBBY

def main(argv):
    try:
        if parseArguments(argv) is False:
            return
    except ValueError as e:
        print(f"Error: {e}")
        return

    window_size = 1000
    current_state: AppState = AppState[GLOBAL_CONFIG.get('mode')]
    try:
        gfx = None
        if current_state != AppState.TRAIN:
            gfx = GraphicsEngine(window_size)
        env = GymnasiumWrapper()
        agent = DQNAgent()

        while current_state != AppState.EXIT:
            if current_state == AppState.TRAIN or gfx is None:
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
                elif current_state in [
                    AppState.HUMAN_PLAYING,
                    AppState.WATCH_TRAIN,
                    AppState.AGENT_PLAYING_EX,
                    AppState.AGENT_PLAYING_EI
                ]:
                    current_state = game_loop(gfx, env, agent, current_state)

    except ValueError as e:
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nGame interrupted by user. Exiting...")


if __name__ == "__main__":
    main(sys.argv[1:])

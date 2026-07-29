from collections import defaultdict, deque
import random

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
                fps = min(GLOBAL_CONFIG.get("fps_cap") + 5, 1000)
                GLOBAL_CONFIG.update('fps_cap', fps)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_m):
                fps = max(GLOBAL_CONFIG.get("fps_cap") - 5, 1000)
                GLOBAL_CONFIG.update('fps_cap', fps)

    return current_action, APP_STATE

def game_loop(gfx: GraphicsEngine, env: GymnasiumWrapper, agent: DQNAgent, current_state: AppState):
    snake_vision = env.reset(GLOBAL_CONFIG.get("board_size"))
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
            action = agent.get_action(snake_vision, explore_mode=True)
        elif current_state in [
            AppState.AGENT_PLAYING_EX,
            AppState.AGENT_PLAYING_EI
        ]:
            action = agent.get_action(snake_vision, explore_mode=(current_state == AppState.AGENT_PLAYING_EX))

        if GLOBAL_CONFIG.get("print_vision"):
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
            agent.train_step(batch_size=GLOBAL_CONFIG.get("batch_size"))

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

BUCKETS = [
    [5, 6, 7, 8, 9, 10, 12, 14, 15],
    [16, 18, 20, 23, 25, 28, 30],
    [32, 35, 40, 45, 50],
    [55, 60, 65, 70, 75],
    [80, 85, 90, 95, 100],
]

def sample_board_size(progress: float) -> tuple[int, int]:

    if progress < 0.2:
        weights = [0.90, 0.08, 0.02, 0.00, 0.00]
    elif progress < 0.4:
        weights = [0.75, 0.18, 0.06, 0.01, 0.00]
    elif progress < 0.6:
        weights = [0.60, 0.22, 0.13, 0.04, 0.01]
    elif progress < 0.8:
        weights = [0.45, 0.25, 0.18, 0.10, 0.02]
    elif progress < 0.9:
        weights = [0.15, 0.20, 0.30, 0.25, 0.10]
    else:
        weights = [0.05, 0.10, 0.35, 0.35, 0.15]
    bucket_index = random.choices(
        range(len(BUCKETS)),
        weights=weights,
        k=1,
    )[0]

    board_size = random.choice(BUCKETS[bucket_index])

    return bucket_index, board_size

def train_game_loop(env: GymnasiumWrapper, agent: DQNAgent):
    episodes = GLOBAL_CONFIG.get("episodes")
    episode = agent.last_episode

    bucket_index, board_size = sample_board_size(episode / episodes)
    snake_vision = env.reset(board_size=board_size)

    print_every = 10000

    global_step = agent.total_steps

    debug = ""
    while episode < episodes:
        global_step += 1

        action = Action(
            agent.get_action(
                snake_vision,
                explore_mode=True,
                bucket_index=bucket_index,
            )
        )

        next_state, reward, is_dead = env.step(action)

        agent.remember(
            snake_vision,
            action,
            next_state,
            reward,
            is_dead,
        )

        if len(agent.memory) < 500_000:
            continue
        
        agent.train_step(
            GLOBAL_CONFIG.get("batch_size"),
            bucket_index=bucket_index,
        )

        if global_step % GLOBAL_CONFIG.get("save_interval") == 0:
            agent.update_target_network()
            
            
        if global_step % print_every == 0:
            debug += (
                f"episode {episode}: "
                f"All Steps: {global_step}, "
                f"Memory Size: {len(agent.memory)}, "
                f"Board Size: {board_size}x{board_size}, "
                f"Epsilon: {agent.get_epsilon(bucket_index):.4f}, "
                f"Loss: "
                f"{agent.losses[-1] if agent.losses else None}\n"
            )

            debug += agent.evaluate(
                env,
                games_per_board=50 if global_step % (print_every*10) == 0 else 5,
            )

            print(debug)
            debug = ""

        if not is_dead:
            snake_vision = next_state
            continue

        episode += 1
        agent.last_episode = episode

        progress = agent.total_steps / GLOBAL_CONFIG.get("total_steps")
        bucket_index, board_size = sample_board_size(progress)

        snake_vision = env.reset(board_size=board_size)

    debug += (
        f"episode {episode}: "
        f"All Steps: {global_step}, "
        f"Memory Size: {len(agent.memory)}, "
        f"Board Size: {board_size}x{board_size}, "
        f"Epsilon: {agent.get_epsilon(bucket_index):.4f}, "
        f"Loss: "
        f"{agent.losses[-1] if agent.losses else None}\n"
    )

    debug += agent.evaluate(
        env,
        games_per_board=20,
    )

    print(debug)
    debug = ""
    if agent.losses:
        diffs = np.diff(agent.losses)
        print(
            f"Training completed after {episodes} episodes. "
            f"Loss diff mean: {diffs.mean():.6f}, "
            f"std: {diffs.std():.6f}"
        )
    else:
        print(f"Training completed after {episodes} episodes.")

    return AppState.LOBBY

def main(argv):
    try:
        if parseArguments(argv) is False:
            return
    except ValueError as e:
        print(f"Error: {e}")
        return

    window_size = 1000
    current_state: AppState = AppState[GLOBAL_CONFIG.get("mode")]
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
                        gfx.refresh()
                elif current_state in [
                    AppState.HUMAN_PLAYING,
                    AppState.WATCH_TRAIN,
                    AppState.AGENT_PLAYING_EX,
                    AppState.AGENT_PLAYING_EI
                ]:
                    current_state = game_loop(gfx, env, agent, current_state)

    except ValueError as e:
        return print(f"Error: {e}")
    except KeyboardInterrupt:
        return print("\nGame interrupted by user. Exiting...")

if __name__ == "__main__":
    main(sys.argv[1:])

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
    [5, 6, 7, 8, 9, 10, 12, 15],
    [16, 18, 20, 24, 27, 30],
    [32, 35, 40, 45, 50],
    [55, 60, 65, 70, 75],
    [80, 85, 90, 95, 100],
]

def sample_board_size(progress: float) -> tuple[int, int]:

    if progress < 0.2:
        weights = [1.0, 0.0, 0.0, 0.0, 0.0]
    elif progress < 0.4:
        weights = [0.7, 0.3, 0.0, 0.0, 0.0]
    elif progress < 0.6:
        weights = [0.5, 0.3, 0.2, 0.0, 0.0]
    elif progress < 0.8:
        weights = [0.35, 0.30, 0.20, 0.15, 0.0]
    else:
        weights = [0.20, 0.20, 0.20, 0.20, 0.20]

    bucket_index = random.choices(
        range(len(BUCKETS)),
        weights=weights,
        k=1,
    )[0]

    board_size = random.choice(BUCKETS[bucket_index])

    return bucket_index, board_size

# def train_game_loop(env: GymnasiumWrapper, agent: DQNAgent):
#     snake_vision = env.reset(GLOBAL_CONFIG.get("board_size"))
#     episodes = GLOBAL_CONFIG.get("episodes")
#     scores_length = 1000
#     print_every = scores_length // 5
#     scores = deque(maxlen=print_every)
#     episode = 0
#     global_step = 0
#     bucket_index = 0
#     debug: str = ""

#     while episode < episodes:
#         global_step += 1

#         raw_action = agent.get_action(snake_vision, explore_mode=True, bucket_index=bucket_index)
#         action = Action(raw_action)

#         if GLOBAL_CONFIG.get("print_vision"):
#             if action is Action.UP:
#                 print("UP")
#             elif action is Action.DOWN:
#                 print("DOWN")
#             elif action is Action.LEFT:
#                 print("LEFT")
#             elif action is Action.RIGHT:
#                 print("RIGHT")

#         next_snake_vision, reward, is_dead = env.step(action)
#         agent.remember(snake_vision, action, next_snake_vision, reward, is_dead)
#         agent.train_step(batch_size=GLOBAL_CONFIG.get("batch_size"), bucket_index=bucket_index)
        
#         if global_step % GLOBAL_CONFIG.get("save_interval") == 0:
#             agent.update_target_network()

#         if is_dead:
#             episode += 1
#             agent.last_episode = episode
#             board_score, green_apple_eaten, red_apple_eaten, _ = env.board.get_board_statistics()
#             scores.append((
#                 board_score,
#                 GLOBAL_CONFIG.get("board_size"),
#                 green_apple_eaten,
#                 red_apple_eaten,
#             ))

#             if episode % print_every == 0 and scores:
#                 per_board = defaultdict(list)

#                 for score, board_size, green, red in scores:
#                     per_board[board_size].append((score, green, red))

#                 debug += "episode {}: Step: {}, Epsilon: {:.4f}, Memory Size: {}, Loss: {}\n".format(
#                     episode,
#                     global_step,
#                     agent.epsilon,
#                     len(agent.memory),
#                     agent.losses[-1] if agent.losses else None,
#                 ) + f"Episode: {episode}\n"

#                 for board_size in sorted(per_board):
#                     games = per_board[board_size]

#                     avg_score = sum(x[0] for x in games) / len(games)
#                     avg_green = sum(x[1] for x in games) / len(games)
#                     avg_red = sum(x[2] for x in games) / len(games)
#                     best_score = max(x[0] for x in games)
#                     success_rate = sum(x[0] > 0 for x in games) / len(games) * 100

#                     debug += (
#                         f"  {board_size:3}x{board_size}: "
#                         f"Avg Score ({len(games)} games): {avg_score:.2f}, "
#                         f"Best: {best_score}, "
#                         f"Success: {success_rate:.1f}%, "
#                         f"Green: {avg_green:.2f}, "
#                         f"Red: {avg_red:.2f}\n"
#                     )

#                 scores.clear()
            
#             progress = agent.last_episode / (agent.last_episode + episodes)
#             bucket_index, board_size = sample_board_size(progress)
#             snake_vision = env.reset(board_size=board_size)
#         else:
#             snake_vision = next_snake_vision
        
#         if episode % 50 == 0 and episode > 0:
#             print(debug, end="")
#             debug = ""

#     if agent.losses:
#         diffs = np.diff(agent.losses)
#         debug += "Training completed after {} episodes. Loss diff mean: {:.6f}, std: {:.6f}\n".format(
#             episodes, diffs.mean(), diffs.std()
#         )
#     else:
#         debug += f"Training completed after {episodes} episodes. No losses recorded.\n"

#     print(debug, end="")

#     return AppState.LOBBY

def train_game_loop(env: GymnasiumWrapper, agent: DQNAgent):
    episodes = GLOBAL_CONFIG.get("episodes")
    episode = agent.last_episode

    bucket_index, board_size = sample_board_size(episode / episodes)
    snake_vision = env.reset(board_size=board_size)

    print_every = 200

    global_step = 0

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

        next_state, reward, done = env.step(action)

        agent.remember(
            snake_vision,
            action,
            next_state,
            reward,
            done,
        )

        agent.train_step(
            GLOBAL_CONFIG.get("batch_size"),
            bucket_index=bucket_index,
        )

        if global_step % GLOBAL_CONFIG.get("save_interval") == 0:
            agent.update_target_network()

        if not done:
            snake_vision = next_state
            continue

        episode += 1
        agent.last_episode = episode

        progress = episode / episodes
        bucket_index, board_size = sample_board_size(progress)

        snake_vision = env.reset(board_size=board_size)

        if episode % print_every == 0:
            debug += (
                f"episode {episode}: "
                f"Step: {global_step}, "
                f"Epsilon: "
                f"{agent.get_epsilon(bucket_index):.4f}, "
                f"Memory Size: {len(agent.memory)}, "
                f"Loss: "
                f"{agent.losses[-1] if agent.losses else None}\n"
            )

            debug += f"Episode: {episode}:\n"

            debug += agent.evaluate(
                env,
                board_sizes=[
                    5, 6, 7, 8, 9, 10,
                    12, 15,
                    16, 18, 20,
                    24, 27, 30,
                    32, 35, 40, 45, 50,
                    55, 60, 65, 70, 75,
                    80, 85, 90, 95, 100,
                ],
                games_per_board=5,
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
        print(f"Error: {e}")
    except KeyboardInterrupt:
        print("\nGame interrupted by user. Exiting...")


if __name__ == "__main__":
    main(sys.argv[1:])

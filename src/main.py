from collections import defaultdict, deque
import random

from gymnasium_wrapper import GymnasiumWrapper
from graphics_engine import GraphicsEngine
from general_parsing import parseArguments
from constants import Action, AppState
from app_config import GLOBAL_CONFIG
from dqn_agent import DQNAgent
from utilize import sample_board_size

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
            fps = GLOBAL_CONFIG.get("fps_cap")
            if event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_p):
                fps = min(fps + 5, 1000)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS, pygame.K_m):
                fps = max(fps - 5, 5)
            GLOBAL_CONFIG.update('fps_cap', fps)

    return current_action, APP_STATE

def get_curriculum_progress(agent: DQNAgent) -> float:
    completed_steps = sum(min(steps, agent.end_total_steps) for steps in agent.bucket_steps)

    return completed_steps / (len(agent.bucket_steps) * agent.end_total_steps)

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

        next_snake_vision, reward, is_dead, _ = env.step(action)
        if current_state == AppState.WATCH_TRAIN:
            agent.remember(0, snake_vision, action, next_snake_vision, reward, is_dead)
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

def train_game_loop(env: GymnasiumWrapper, eval_env: GymnasiumWrapper, agent: DQNAgent):
    if min(agent.bucket_steps) >= agent.end_total_steps:
        return AppState.LOBBY

    board_stats = sample_board_size(agent.bucket_steps, agent.balancing_steps)
    if board_stats is None:
        return AppState.LOBBY

    bucket_index, board_size, balancing_phase = board_stats
    snake_vision = env.reset(board_size=board_size)
    episode = agent.last_episode
    
    print_every = 200_000
    printing = True

    target_update_every = GLOBAL_CONFIG.get("target_update_every")
    # checkpoint_every = GLOBAL_CONFIG.get("checkpoint_every")
    train_every = GLOBAL_CONFIG.get("train_every")
    batch_size = GLOBAL_CONFIG.get("batch_size")

    debug = ""
    # save_now = False
    while True:
        agent.env_steps += 1

        action = Action(
            agent.get_action(
                snake_vision,
                explore_mode=True,
                bucket_index=bucket_index,
            )
        )

        next_state, reward, is_dead, stop_training = env.step(action, training=True)
        agent.remember(
            bucket_index,
            snake_vision,
            action,
            next_state,
            reward,
            is_dead,
        )

        if agent.env_steps % train_every == 0:
            loss = agent.train_step(batch_size, bucket_index, balancing_phase)

            if loss is not None:
                if agent.total_steps % target_update_every == 0:
                    agent.update_target_network()

        # if agent.env_steps % checkpoint_every == 0:
        #     save_now = True

        if agent.env_steps % print_every == 0 and printing:
            debug += (
                f"episode {episode}: "
                f"All Steps: {agent.env_steps} ({get_curriculum_progress(agent):.4f}), "
                f"Bucket Steps: ({', '.join(str(steps) for steps in agent.bucket_steps)}), "
                f"Balancing Steps: {agent.balancing_steps}, "
                f"Memory Size: ({', '.join(str(len(bucket)) for bucket in agent.memory)}), "
                f"Loss: "
                f"{agent.losses[-1]:.6f}\n" if agent.losses else "None\n"
            )
            debug += agent.get_training_distribution()

            debug += agent.evaluate(eval_env, games_per_board=10, save_checkpoint=True)
            print(debug)
            debug = ""

        if not (is_dead or stop_training):
            snake_vision = next_state
            continue

        episode += 1
        agent.last_episode = episode

        # if save_now:
        #     agent.save_model(f"models/checkpoint_{agent.last_episode}.pth")
        #     save_now = False

        board_stats = sample_board_size(agent.bucket_steps, agent.balancing_steps)
        if board_stats is None:
            break
        bucket_index, board_size, balancing_phase = board_stats
        snake_vision = env.reset(board_size=board_size)

    if printing:
        debug += (
            f"episode {episode}: "
            f"All Steps: {agent.env_steps}, "
            f"Bucket Steps: {', '.join(str(steps) for steps in agent.bucket_steps)}, "
            f"Balancing Steps: {agent.balancing_steps}, "
            f"Memory Size: {', '.join(str(len(bucket)) for bucket in agent.memory)}, "
            f"Board Size: {board_size}x{board_size}, "
            f"Epsilon: {agent.get_epsilon(bucket_index):.4f}, "
            f"Loss: "
            f"{agent.losses[-1] if agent.losses else None}\n"
        )

        debug += agent.evaluate(
            eval_env,
            games_per_board=20,
        )

        print(debug)
        debug = ""
        if agent.losses:
            recent_losses = agent.losses[-1000:]
            print(
                f"Training completed after {episode} episodes. "
                f"Recent Loss Mean: {np.mean(recent_losses):.6f}, "
                f"Recent Loss Std: {np.std(recent_losses):.6f}"
            )
        else:
            print(f"Training completed after {episode} episodes.")

    agent.save_model(f"models/final_checkpoint_{agent.last_episode}.pth")
    return AppState.LOBBY

def main(argv):
    try:
        if parseArguments(argv) is False:
            return
    except ValueError as e:
        print(f"Error: {e}")
        return

    window_size = 1500
    current_state: AppState = AppState[GLOBAL_CONFIG.get("mode")]
    try:
        gfx = None
        if current_state not in [AppState.TRAIN, AppState.EVALUATE]:
            gfx = GraphicsEngine(window_size)
        env = GymnasiumWrapper()
        eval_env = GymnasiumWrapper()
        agent = DQNAgent()

        while current_state != AppState.EXIT:
            if current_state == AppState.TRAIN:
                current_state = train_game_loop(env, eval_env, agent)
                if gfx is None:
                    gfx = GraphicsEngine(window_size)
            elif current_state == AppState.EVALUATE:
                print(agent.evaluate(eval_env, games_per_board=GLOBAL_CONFIG.get("evaluate_model")))
                break
            elif gfx is not None:
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

    agent.save_model()

if __name__ == "__main__":
    main(sys.argv[1:])

from gymnasium_wrapper import GymnasiumWrapper
from graphics_engine import GraphicsEngine
from general_parsing import parseArguments
from constants import Action, AppState
from utilize import sample_board_size
from app_config import GLOBAL_CONFIG
from dqn_agent import DQNAgent
from pathlib import Path

import numpy as np
import pygame
import sys


def get_input(
    current_action: Action,
    APP_STATE: AppState
) -> tuple[Action, AppState]:
    for event in pygame.event.get():
        if (
            event.type == pygame.QUIT or
            (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
        ):
            return current_action, AppState.EXIT

        if (
            APP_STATE == AppState.HUMAN_PLAYING
            and event.type == pygame.KEYDOWN
        ):
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
    completed_steps = (
        sum(min(steps, agent.end_total_steps)
            for steps in agent.bucket_steps)
    )

    return completed_steps / (len(agent.bucket_steps) * agent.end_total_steps)


def game_loop(
    gfx: GraphicsEngine,
    env: GymnasiumWrapper,
    agent: DQNAgent,
    current_state: AppState
) -> AppState:
    """
    Main game loop for human or agent play.
    Args:
        gfx: GraphicsEngine instance for rendering.
        env: GymnasiumWrapper instance for the Snake environment.
        agent: DQNAgent instance for AI control.
        current_state: Current application state.
    Returns:
        Updated application state after the game loop.
    """
    snake_vision = env.reset(GLOBAL_CONFIG.get("board_size"))
    action: Action = Action.RIGHT

    while current_state in [
        AppState.HUMAN_PLAYING,
        AppState.WATCH_TRAIN,
        AppState.AGENT_PLAYING_EX,
        AppState.AGENT_PLAYING_EI
    ]:
        action, current_state = get_input(
            current_action=action,
            APP_STATE=current_state
        )
        if current_state == AppState.EXIT:
            break

        if current_state == AppState.WATCH_TRAIN:
            action = agent.get_action(snake_vision, explore_mode=True)
        elif current_state in [
            AppState.AGENT_PLAYING_EX,
            AppState.AGENT_PLAYING_EI
        ]:
            action = agent.get_action(
                snake_vision,
                explore_mode=(current_state == AppState.AGENT_PLAYING_EX)
            )

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
            agent.remember(
                0,
                snake_vision,
                action,
                next_snake_vision,
                reward,
                is_dead
            )
            agent.train_step(batch_size=GLOBAL_CONFIG.get("batch_size"))

        if current_state in [
            AppState.HUMAN_PLAYING,
            AppState.WATCH_TRAIN,
            AppState.AGENT_PLAYING_EX,
            AppState.AGENT_PLAYING_EI
        ]:
            board_score, green_apple_eaten, red_apple_eaten, board = (
                env.board.get_board_statistics()
            )
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


def one_session_train(
    env: GymnasiumWrapper,
    agent: DQNAgent,
    board_size: int,
    bucket_index: int,
    balancing_phase: bool = False,
    curriculum_training: bool = False
) -> None:
    """
    Train exactly one Snake episode.

    The caller decides:
        - Which board size to use.
        - Which replay bucket to train.
        - Whether this is a balancing phase.
        - How many episodes to train overall.
        - When checkpoints are saved.

    An episode ends on genuine death or training truncation.
    """
    snake_vision = env.reset(board_size=board_size)

    target_update_every = GLOBAL_CONFIG.get("target_update_every")
    train_every = GLOBAL_CONFIG.get("train_every")
    batch_size = GLOBAL_CONFIG.get("batch_size")

    while True:
        agent.env_steps += 1

        action = Action(
            agent.get_action(
                snake_vision,
                explore_mode=True,
                bucket_index=bucket_index
            )
        )

        next_state, reward, is_dead, stop_training = env.step(
            action,
            training=True,
        )
        if stop_training:
            return

        agent.remember(
            bucket_index,
            snake_vision,
            action,
            next_state,
            reward,
            is_dead
        )

        if agent.env_steps % train_every == 0:
            loss = agent.train_step(
                batch_size,
                bucket_index,
                balancing_phase
            )

            if loss is not None:
                if agent.total_steps % target_update_every == 0:
                    agent.update_target_network()

        if is_dead:
            return

        snake_vision = next_state


def train_game_loop(
    env: GymnasiumWrapper,
    eval_env: GymnasiumWrapper,
    agent: DQNAgent
) -> AppState:
    """
    Train using one of two modes:

    Session-based:
        Train N episodes on a fixed 10x10 board.
        Save checkpoints at sessions 1, 10, 100
        (when reached), and at the end.

    Curriculum-based:
        Train using the board-size curriculum until
        sample_board_size() reports completion.
        Save only the final checkpoint.

    Both modes perform one final evaluation.
    """
    training_sessions = GLOBAL_CONFIG.get("training_sessions")
    output_dir = Path(GLOBAL_CONFIG.get("new_weights"))
    save_weights = GLOBAL_CONFIG.get("save_weights")
    agent.session_episodes_completed = 0
    final_path = None

    if save_weights:
        output_dir.mkdir(parents=True, exist_ok=True)

    completed_sessions = 0
    if training_sessions is not None:
        if (not isinstance(training_sessions, int) or training_sessions <= 0):
            raise ValueError(
                "training_sessions must be a positive integer or None."
            )

        bucket_index = 0
        board_size = GLOBAL_CONFIG.get("board_size")

        for session in range(1, training_sessions + 1):
            one_session_train(
                env=env,
                agent=agent,
                board_size=board_size,
                bucket_index=bucket_index,
                balancing_phase=False,
                curriculum_training=False
            )

            agent.last_episode += 1
            completed_sessions += 1
            agent.session_episodes_completed += 1
            if (
                save_weights
                and session in (1, 10, 100, 1000, 10_000, 100_000)
            ):
                agent.save_model(
                    str(output_dir / f"snake_sessions_{session}.pth")
                )

        if save_weights:
            final_path = output_dir / "snake_sessions_final.pth"

    else:
        while True:
            board_stats = sample_board_size(
                agent.bucket_steps,
                agent.balancing_steps
            )
            if board_stats is None:
                break

            bucket_index, board_size, balancing_phase = board_stats
            one_session_train(
                env=env,
                agent=agent,
                board_size=board_size,
                bucket_index=bucket_index,
                balancing_phase=balancing_phase,
                curriculum_training=True
            )

            agent.last_episode += 1
            completed_sessions += 1
            agent.session_episodes_completed += 1

        if save_weights:
            final_path = output_dir / "snake_curriculum_final.pth"

    if save_weights and final_path is not None:
        agent.save_model(str(final_path))

    print(f"\nTraining completed after {completed_sessions:,} sessions.")
    print(f"Total environment steps: {agent.env_steps:,}")
    print(f"Total optimizer updates: {agent.total_steps:,}")

    if agent.losses:
        recent_losses = agent.losses[-1000:]
        print(
            f"Recent Loss Mean: {np.mean(recent_losses):.6f}, "
            f"Recent Loss Std: {np.std(recent_losses):.6f}"
        )

    print("\nFinal Evaluation:\n")
    if training_sessions is not None:
        evaluation = agent.evaluate(
            eval_env,
            games_per_board=20,
            force_board_size=10
        )
    else:
        evaluation = agent.evaluate(eval_env, games_per_board=20)
    print(evaluation)

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
                if GLOBAL_CONFIG.get("training_sessions") is not None:
                    print(agent.evaluate(
                        eval_env,
                        games_per_board=GLOBAL_CONFIG.get("evaluate_model"),
                        force_board_size=10
                    ))
                else:
                    print(agent.evaluate(
                        eval_env,
                        games_per_board=GLOBAL_CONFIG.get("evaluate_model")
                    ))
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


if __name__ == "__main__":
    main(sys.argv[1:])

"""
Example
"""
import time

import torch

WANDB_USAGE = False
WANDB_NOTEBOOK_NAME = "tfg-wero-lidia"

if __name__ == "__main__":
    import random
    random.seed(42)
    from jsonargparse import ArgumentParser, ActionConfigFile
    from atcenv import Environment, rl
    from tqdm import tqdm
    import collections
    from atcenv.definitions import *
    from atcenv.policy import *
    import numpy as np

    MAX_MEMORY_SIZE = 100
    BATCH_SIZE = 32
    GAMMA = 0.95
    TAU = 1
    LR = 0.1
    EXPLORATION_MAX = 0.9
    EXPLORATION_MIN = 0.1
    EXPLORATION_DECAY = 0.00001
    TRAINING_EPISODES = 300
    HIDDEN_NEURONS = 256
    TARGET_UPDATE = 10
    RENDERING_FREQUENCY = 1
    SHORT_MEMORY_SIZE = 2
    render = True

    if WANDB_USAGE:
        import wandb
        wandb.init(project="dqn", entity="tfg-wero-lidia",
                   name='testing new graph')

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(Environment, 'env')

    # parse arguments
    args = parser.parse_args()

    # init environment
    env = Environment(**vars(args.env))
    comparison_env = Environment(**vars(args.env))

    # Prioritized replay buffer
    Experience = collections.namedtuple('Experience', field_names=['states', 'actions', 'rewards', 'next_states'])
    if WANDB_USAGE:
        wandb.config.update({"max_memory_size": MAX_MEMORY_SIZE, "batch_size": BATCH_SIZE, "gamma": GAMMA, "tau": TAU,
                             "lr": LR, "exploration_max": EXPLORATION_MAX, "exploration_min": EXPLORATION_MIN,
                             "exploration_decay": EXPLORATION_DECAY, "training_episodes": TRAINING_EPISODES,
                             "hidden_neurons": HIDDEN_NEURONS, "n_neighbours": env.n_neighbours})

    do_nothing = [0] * comparison_env.num_flights
    replay_buffer = rl.ReplayBuffer(MAX_MEMORY_SIZE)
    short_memo = rl.ReplayBuffer(SHORT_MEMORY_SIZE)

    print('\n*** Filling the Replay Buffer ***')
    for e in tqdm(range(MAX_MEMORY_SIZE//env.num_flights)):
        obs = env.reset()
        done = False
        short_exp = Experience(env.distances_matrix(), do_nothing, do_nothing, do_nothing)
        short_memo.append(short_exp)
        while not done:
            previous_distances = env.distances_matrix()
            actions = policy_action(obs, short_memo, env)
            rew, next_obs, done = env.step(actions)
            for i in range(0, env.num_flights):
                if i not in env.done:
                    if len(replay_buffer.buffer) < MAX_MEMORY_SIZE:
                        experience = Experience(obs[i], actions[i], rew[i], next_obs[i])
                        replay_buffer.append(experience)

            short_exp = Experience(previous_distances, actions, do_nothing, do_nothing)
            short_memo.append(short_exp)
            obs = next_obs
        env.close()

    print('\n*** Replay Buffer is now full ***')
    dqn = rl.DQN(MAX_MEMORY_SIZE, BATCH_SIZE, GAMMA, TAU, LR, EXPLORATION_MAX, EXPLORATION_MIN, EXPLORATION_DECAY, env,
                 replay_buffer, HIDDEN_NEURONS, target_update=TARGET_UPDATE)

    print('\n Training the DQN model with experience.')
    for e in tqdm(range(TRAINING_EPISODES)):
        dqn.learn()

    print('\n DQN training while adding new experiences.')
    for e in tqdm(range(args.episodes)):
        obs = env.reset()
        c_obs = comparison_env.reset()
        done = False
        rew_episode = 0
        while not done:
            previous_distances = env.distances_matrix()
            actions = []
            for i in range(env.num_flights):
                action = dqn.select_action(obs)
                actions.append(action)
            rew, next_obs, done = env.step(actions)
            rew_episode += np.average(rew)
            c_rew, c_obs, c_done = comparison_env.step(do_nothing)

            for i in range(0, env.num_flights):
                if i not in env.done:
                    experience = Experience(obs[i], actions[i], rew[i], next_obs[i])
                    replay_buffer.append(experience)

            if len(replay_buffer.buffer) > MAX_MEMORY_SIZE:
                dqn.learn()
            short_exp = Experience(previous_distances, actions, do_nothing, do_nothing)
            short_memo.append(short_exp)
            obs = next_obs

            if (e % RENDERING_FREQUENCY == 0 or e == args.episodes-1) and render:
                env.render()
                time.sleep(0.01)

            if WANDB_USAGE:
                wandb.log({'conflicts/step with policy': env.n_conflicts_step,
                           'conflicts/step without policy': comparison_env.n_conflicts_step,
                           'n_aircraft in conflict/step': len(env.conflicts),
                           'exploration rate': dqn.exploration_rate,
                           'avg rew/step': np.average(rew)})
        if WANDB_USAGE:
            n_real_conflicts_episode = 0
            n_real_conflicts_episode_without_policy = 0
            for i in range(env.num_flights):
                for j in range(env.num_flights):
                    if env.matrix_real_conflicts_episode[i, j]:
                        n_real_conflicts_episode += 1
                    if comparison_env.matrix_real_conflicts_episode[i, j]:
                        n_real_conflicts_episode_without_policy += 1
            wandb.log({'conflicts/episode with policy': env.n_conflicts_episode,
                       'conflicts/episode without policy': comparison_env.n_conflicts_episode,
                       'avg rew/episode': rew_episode,
                       'extra distance': sum(env.flights[i].actual_distance-env.flights[i].planned_distance for i in
                                             range(env.num_flights)),
                       'real conflicts/episode': n_real_conflicts_episode_without_policy})
        env.close()
        comparison_env.close()

    print('\nSaving the model')
    PATH = f'./target_net/eps_dqn{args.episodes}_eps_policy_{TRAINING_EPISODES}_hidden_n_{HIDDEN_NEURONS}'
    torch.save(dqn.target_net.state_dict(), PATH)

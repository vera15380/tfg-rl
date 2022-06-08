"""
Example
"""
import time
import torch

WANDB_USAGE = True
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

    MAX_MEMORY_SIZE = 1000
    BATCH_SIZE = 100
    GAMMA = 0.95
    TAU = 0.8
    LR = 1e-4
    EPSILON = 0.5
    TRAINING_EPISODES = 30000
    HIDDEN_NEURONS = 128
    TARGET_UPDATE = 100
    RENDERING_FREQUENCY = 50
    SHORT_MEMORY_SIZE = 2
    render = True
    reward_type = "with_alert"

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=500)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(Environment, 'env')

    # parse arguments
    args = parser.parse_args()

    # init environment
    env = Environment(**vars(args.env))
    comparison_env = Environment(**vars(args.env))

    # Prioritized replay buffer
    Experience = collections.namedtuple('Experience', field_names=['states', 'actions', 'rewards', 'next_states'])

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
    dqn = rl.DQN(MAX_MEMORY_SIZE, BATCH_SIZE, GAMMA, TAU, LR, EPSILON, env, replay_buffer, HIDDEN_NEURONS,
                 max_episodes=args.episodes, target_update=TARGET_UPDATE)

    print('\n Training the DQN model with experience.')
    for e in tqdm(range(TRAINING_EPISODES)):
        dqn.learn()

    if WANDB_USAGE:
        import wandb
        wandb.init(project="dqn", entity="tfg-wero-lidia",
                   name="tau=0.8 eps=0.5 gamma=0.95 lr=1e-4 neurons=128 angle change=5º eps=500 n_actions= 8 w/ alert sector obs tgt updt 100 sector correction v2")

        wandb.config.update({"max_memory_size": MAX_MEMORY_SIZE, "batch_size": BATCH_SIZE, "gamma": GAMMA, "tau": TAU,
                             "lr": LR, "exploration_max": EPSILON, "MAX_EPISODES": args.episodes,
                             "exploration_decay": EPSILON, "training_episodes": TRAINING_EPISODES,
                             "hidden_neurons": HIDDEN_NEURONS, "n_neighbours": env.n_neighbours, "angle_change":
                                 env.angle_change, "n_actions": env.num_discrete_actions, "rew_type": reward_type})

    print('\n DQN training while adding new experiences.')
    for e in tqdm(range(args.episodes)):
        n_turns_episode = 0
        obs = env.reset()
        c_obs = comparison_env.reset()
        done = False
        rew_episode = 0
        while not done:
            loss = np.NaN
            previous_distances = env.distances_matrix()
            actions = []
            for i in range(env.num_flights):
                action = dqn.select_action(obs, e)
                actions.append(action)
            rew, next_obs, done = env.step(actions)
            rew_episode += np.average(rew)
            c_rew, c_obs, c_done = comparison_env.step(do_nothing)

            for i in range(0, env.num_flights):
                if i not in env.done:
                    experience = Experience(obs[i], actions[i], rew[i], next_obs[i])
                    replay_buffer.append(experience)

            if len(replay_buffer.buffer) > MAX_MEMORY_SIZE:
                loss = dqn.learn()
            short_exp = Experience(previous_distances, actions, do_nothing, do_nothing)
            short_memo.append(short_exp)
            obs = next_obs

            if (e % RENDERING_FREQUENCY == 0 or e == args.episodes-1) and render:
                env.render()
                time.sleep(0.01)

            if WANDB_USAGE:
                n_turns_step = sum(env.flights[i].n_turns for i in range(env.num_flights))
                n_turns_episode += n_turns_step
                if env.i == env.max_episode_len:
                    distance_left_to_target = sum(env.flights[i].position.distance(env.flights[i].target) for i in
                                                  range(env.num_flights))
                wandb.log({'[CONFLICTS] - conflicts/step with policy': env.n_conflicts_step,
                           '[CONFLICTS] - conflicts/step without policy': comparison_env.n_conflicts_step,
                           '[CONFLICTS] - n_aircraft in conflict/step': len(env.conflicts),
                           '[REWARD] - avg rew/step': np.average(rew),
                           '[EXTRA] - n_turns taken/step': n_turns_step})

        if WANDB_USAGE:
            n_real_conflicts_episode = 0
            n_real_conflicts_episode_without_policy = 0
            distance_left_to_target = 0
            for i in range(env.num_flights):
                for j in range(env.num_flights):
                    if env.matrix_real_conflicts_episode[i, j]:
                        n_real_conflicts_episode += 1
                    if comparison_env.matrix_real_conflicts_episode[i, j]:
                        n_real_conflicts_episode_without_policy += 1
            wandb.log({'[CONFLICTS] - conflicts/episode with policy': env.n_conflicts_episode,
                       '[CONFLICTS] - conflicts/episode without policy': comparison_env.n_conflicts_episode,
                       '[REWARD] - avg rew/episode': rew_episode,
                       '[EXTRA] - extra distance': sum(env.flights[i].actual_distance-env.flights[i].planned_distance
                                                       for i in range(env.num_flights)),
                       '[EXTRA] - n_turns taken/episode': n_turns_episode,
                       '[CONFLICTS] - real conflicts/episode': n_real_conflicts_episode_without_policy,
                       '[EXTRA] - distance left to target/episode': distance_left_to_target})
        env.close()
        comparison_env.close()

    print('\nSaving the model')
    PATH = f'./target_net/eps_dqn{args.episodes}_eps_{EPSILON}_policy_{TRAINING_EPISODES}_hidden_n_{HIDDEN_NEURONS}' \
           f'_angle_{env.angle_change}_lr_{LR}_gamma_{GAMMA}_tau_{TAU}_time_{time.time()}_n_actions' \
           f'_{env.num_discrete_actions}'
    torch.save(dqn.target_net.state_dict(), PATH)

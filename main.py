import time

import numpy as np
import torch
import beepy
import random

from jsonargparse import ArgumentParser, ActionConfigFile
from torch.utils.tensorboard import SummaryWriter

from atcenv import Environment, rl, wandb_graphs
from tqdm import tqdm
import collections
from atcenv.definitions import *
from atcenv.policy import *

WANDB_USAGE = True
WANDB_NOTEBOOK_NAME = "emc_upc"
random.seed(5)
array = [0.1]
if __name__ == "__main__":
    for GAMMA in array:
        # parameter definition
        EPISODES = 1000
        EVALUATION_EPISODES = 100
        MAX_MEMORY_SIZE = 100
        BATCH_SIZE = 64
        # GAMMA = 0.75
        TAU = 0.1
        LR = 0.001
        EXPLORATION_MAX = 1
        EXPLORATION_MIN = 0.1
        EXPLORATION_DECAY = 0.00001
        TRAINING_EPISODES = 500
        HIDDEN_NEURONS = 128
        TARGET_UPDATE = 10
        RENDERING_FREQUENCY = 1
        render = False
        CLOSING = False
        POLICY = False
        reward_type = "conflict -10 other actions"

        parser = ArgumentParser(
            prog='Conflict resolution environment',
            description='Basic conflict resolution environment for training policies with reinforcement learning',
            print_config='--print_config',
            parser_mode='yaml'
        )
        parser.add_argument('--episodes', type=int, default=EPISODES)
        parser.add_argument('--config', action=ActionConfigFile)
        parser.add_class_arguments(Environment, 'env')
        # parse arguments
        args = parser.parse_args()

        # init environment
        env = Environment(**vars(args.env))
        comparison_env = Environment(**vars(args.env))

        # Replay buffer
        Experience = collections.namedtuple('Experience', field_names=['states', 'actions', 'rewards', 'next_states'])
        short_memo = rl.ReplayBuffer(2)
        do_nothing = [0] * comparison_env.num_flights
        replay_buffer = rl.ReplayBuffer(MAX_MEMORY_SIZE)

        # dqn initialization
        dqn = rl.DQN(MAX_MEMORY_SIZE, BATCH_SIZE, GAMMA, TAU, LR, EXPLORATION_MAX, EXPLORATION_MIN, EXPLORATION_DECAY,
                     env,
                     replay_buffer, HIDDEN_NEURONS, max_episodes=args.episodes, target_update=TARGET_UPDATE)
        writer_policy = SummaryWriter(comment='policy')
        if WANDB_USAGE:
            import wandb

            run = wandb.init(project="hyperparameter_tuning", entity="emc-upc",
                             name=f"random seed", reinit=True)

            wandb.config.update({"max_memory_size": MAX_MEMORY_SIZE, "batch_size": BATCH_SIZE, "gamma": GAMMA,
                                 "tau": TAU,
                                 "lr": LR, "exploration_max": EXPLORATION_MAX, "MAX_EPISODES": args.episodes,
                                 "exploration_min": EXPLORATION_MIN, "exploration_decay": EXPLORATION_DECAY,
                                 "training_episodes": TRAINING_EPISODES, "hidden_neurons": HIDDEN_NEURONS,
                                 "angle_change": env.angle_change,
                                 "n_actions": env.num_discrete_actions,
                                 "rew_type": reward_type,
                                 "alert_dist": env.alert_distance,
                                 "target_updt": TARGET_UPDATE, "dr": env.detection_range,
                                 "detection_dist": env.detection_range,
                                 "n_flights": env.num_flights}, allow_val_change=True)
        if POLICY:
            # policy application
            print('\n*** TRAINING WITH HUMAN POLICY ***')
            cumulative_reward = 0
            successful_rate_list = []
            loss = np.NaN
            for e in tqdm(range(TRAINING_EPISODES)):
                n_turns_episode = 0
                rew_episode = 0
                obs, _ = env.reset()
                c_obs = comparison_env.comparison_reset(_)
                done = False
                short_exp = Experience(env.distances_matrix(), do_nothing, do_nothing, do_nothing)
                short_memo.append(short_exp)
                while not done:
                    previous_distances = env.distances_matrix()
                    actions = policy_action(short_memo, env)
                    rew, next_obs, done = env.step(actions)
                    c_rew, c_obs, c_done = comparison_env.step(do_nothing)
                    for i in range(0, env.num_flights):
                        if i not in env.done:
                            experience = Experience(obs[i], actions[i], rew[i], next_obs[i])
                            replay_buffer.append(experience)
                    # env.render()
                    # time.sleep(0.03)
                    if len(replay_buffer.buffer) >= MAX_MEMORY_SIZE:
                        # if buffer size is enough, learn.
                        loss = dqn.learn()

                    short_exp = Experience(previous_distances, actions, do_nothing, do_nothing)
                    short_memo.append(short_exp)
                    if WANDB_USAGE:
                        n_turns_step = sum(env.flights[i].n_turns for i in range(env.num_flights))
                        n_turns_episode += n_turns_step
                        rew_average = wandb_graphs.wandb_per_step(writer_policy, env, comparison_env, n_turns_step, rew,
                                                                  loss, dqn)
                        rew_episode += rew_average
                    obs = next_obs

                if WANDB_USAGE:
                    cumulative_reward += rew_episode
                    wandb_graphs.wandb_per_episode(writer_policy, env, comparison_env, rew_episode, n_turns_episode,
                                                   successful_rate_list,
                                                   e, cumulative_reward, CLOSING)
                env.close()

        print('\n *** DQN training while adding new experiences ***')
        # training the model with new experiences from reinforcement learning
        successful_rate_perc = dqn.train_model(env, comparison_env, EPISODES, RENDERING_FREQUENCY, render, CLOSING)

        # saving the trained target neural network
        print('\n *** Saving the model ***')
        PATH = f'./target_net/eps_dqn{args.episodes}_expmin_{dqn.exploration_min}_policy_{TRAINING_EPISODES}_hidden_n_'\
               f'{HIDDEN_NEURONS}' \
               f'_angle_{env.angle_change}_lr_{LR}_gamma_{GAMMA}_tau_{TAU}_time_{time.time()}_n_actions' \
               f'_{env.num_discrete_actions}_1M'
        torch.save(dqn.target_net.state_dict(), PATH)

        if WANDB_USAGE:
            wandb.config.update({"success_rate of episodes": successful_rate_perc})

        successful_rate_list = []
        successful_rate_perc_eval = 0
        rew_history = []
        conf_history_ep = []
        real_conf_history = []
        min_distance_history = []
        n_turns_history = []
        extra_distance_history = []
        dist_to_target_history = []
        # starting evaluation
        for eval in tqdm(range(EVALUATION_EPISODES)):
            beepy.beep(sound="ready")
            dqn.exploration_rate = 0
            obs, _ = env.reset()
            successful_rate = 0
            done = False
            n_turns_episode = 0
            rew_episode = 0
            while not done:
                n_turns_step = 0
                rew_step = 0
                actions = []
                for i in range(env.num_flights):
                    action = dqn.select_action(obs, eval)
                    actions.append(action)
                rew, next_obs, done = env.step(actions)
                rew_without_nan = [x for x in rew if np.isnan(x) == False]
                if len(rew_without_nan) != 0:
                    rew_average_step = np.average(rew_without_nan)
                else:
                    rew_average_step = 0
                rew_episode += rew_average_step
                for i in range(0, env.num_flights):
                    if i not in env.done:
                        experience = Experience(obs[i], actions[i], rew[i], next_obs[i])
                        replay_buffer.append(experience)

                n_turns_step = sum(env.flights[i].n_turns for i in range(env.num_flights))
                n_turns_episode += n_turns_step

                obs = next_obs
                # env.render()
                # time.sleep(0.1)

            if env.n_conflicts_episode == 0:
                successful_rate_list.append(1)
            else:
                successful_rate_list.append(0)

            successful_rate_perc_eval = (sum(successful_rate_list) / len(successful_rate_list)) * 100
            rew_history.append(rew_episode)
            conf_history_ep.append(env.n_conflicts_episode)
            n_real_conflicts_episode = 0
            for i in range(env.num_flights):
                for j in range(env.num_flights):
                    if env.matrix_real_conflicts_episode[i, j]:
                        n_real_conflicts_episode += 1
            real_conf_history.append(n_real_conflicts_episode)
            min_distance_history.append(min(env.critical_distance))
            n_turns_history.append(n_turns_episode)
            extra_distance_history.append(sum(u.m * (env.flights[i].actual_distance - env.flights[i].planned_distance)
                                              for i in range(env.num_flights)))
            dist_to_target_history.append(sum(env.flights[i].position.distance(env.flights[i].target) for i in
                                              range(env.num_flights)))
            env.close()
            comparison_env.close()

        if WANDB_USAGE:
            wandb.config.update({"success_rate_eval of episodes": successful_rate_perc_eval,
                                 "avg conf/episode": np.average(conf_history_ep),
                                 "avg real conf/episode": np.average(real_conf_history),
                                 "avg rew/episode": np.average(rew_history),
                                 "avg min distance/ep": np.average(min_distance_history),
                                 "avg n_turns/ep": np.average(n_turns_history),
                                 "avg extra dist/ep": np.average(extra_distance_history),
                                 "avg dist to target/ep": np.average(dist_to_target_history)})
            run.finish()

        print(f"The success rate was of: {successful_rate_perc} %")
        print(f"The success rate in evaluation was of: {successful_rate_perc_eval} %")

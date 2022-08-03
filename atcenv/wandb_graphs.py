import numpy as np
import wandb
from atcenv.definitions import *
from torch.utils.tensorboard import SummaryWriter


def wandb_per_step(writer, env, comparison_env, n_turns_step, rew, loss, dqn):
    distance_left_to_target = 0
    rew_without_nan = [x for x in rew if np.isnan(x) == False]

    if len(rew_without_nan) != 0:
        rew_average = np.average(rew_without_nan)
    else:
        rew_average = 0

    wandb.log({'1.1.- conflicts/step': env.n_conflicts_step,
               '1.2.- conflicts/step without DQN': comparison_env.n_conflicts_step,
               '1.3.- n_aircraft in conflict/step': len(env.conflicts),
               '1.4.- avg rew/step': rew_average,
               '1.5.- n_turns taken/step': n_turns_step,
               '1.6.- exploration rate': dqn.exploration_rate,
               '1.7.- DQN loss': loss})
    return rew_average


def wandb_per_episode(writer, env, comparison_env, rew_episode, n_turns_episode, successful_rate_list, episode,
                      cumulative_reward, CLOSING):
    successful_rate = 0
    distance_left_to_target = 0
    if (env.i >= env.max_episode_len and CLOSING) or env.n_conflicts_episode == 0:
        successful_rate = 1

    if env.i >= env.max_episode_len:
        distance_left_to_target = sum(env.flights[i].position.distance(env.flights[i].target) for i in
                                      range(env.num_flights))
    successful_rate_list.append(successful_rate)
    n_real_conflicts_episode = 0
    n_real_conflicts_episode_without_policy = 0
    for i in range(env.num_flights):
        for j in range(env.num_flights):
            if env.matrix_real_conflicts_episode[i, j]:
                n_real_conflicts_episode += 1
            if comparison_env.matrix_real_conflicts_episode[i, j]:
                n_real_conflicts_episode_without_policy += 1
    wandb.log({'2.1.- conflicts/episode': env.n_conflicts_episode,
               '2.2.- conflicts/episode without DQN': comparison_env.n_conflicts_episode,
               '2.3.- conflicts/episode difference': comparison_env.n_conflicts_episode -
                                                             env.n_conflicts_episode,
               '2.4.- avg rew/episode': rew_episode,
               '2.5.- extra distance': sum(u.m * (env.flights[i].actual_distance - env.flights[i].planned_distance)
                                               for i in range(env.num_flights)),
               '2.6.-  n_turns taken/episode': n_turns_episode,
               '2.7.- real conflicts/episode without DQN': n_real_conflicts_episode_without_policy,
               '2.8.- real conflicts/episode': n_real_conflicts_episode,
               '2.9.- distance left to target/episode': distance_left_to_target * u.m,
               '2.10.- Minimum separation distance/episode': min(env.critical_distance),
               '2.11.- Minimum separation distance/episode without DQN': min(comparison_env.critical_distance),
               '2.12.- Episode length': env.i,
               '2.13.- Successful episode (Y/N)': successful_rate,
               '2.14.- Cumulative reward/episode': cumulative_reward})

    writer.add_scalar('1_num_conflicts/episode', env.n_conflicts_episode, episode)
    writer.add_scalar('2_conflicts/episode_without_DQN', comparison_env.n_conflicts_episode, episode)
    writer.add_scalar('3_conflicts/episode_difference', comparison_env.n_conflicts_episode - env.n_conflicts_episode,
                      episode)
    writer.add_scalar('4_avg_rew/episode', rew_episode, episode)
    writer.add_scalar('5_extra_distance/episode', sum(u.m * (env.flights[i].actual_distance -
                                                             env.flights[i].planned_distance) for i in
                                                      range(env.num_flights)), episode)
    writer.add_scalar('6_real_conflicts/episode', n_real_conflicts_episode, episode)
    writer.add_scalar('7_real_conflicts/episode_without_DQN', n_real_conflicts_episode_without_policy, episode)
    writer.add_scalar('8_distance left to target/episode', distance_left_to_target * u.m, episode)
    writer.add_scalar('9_Minimum separation distance/episode', min(env.critical_distance), episode)
    writer.add_scalar('10_Minimum separation distance/episode without DQN', min(comparison_env.critical_distance),
                      episode)

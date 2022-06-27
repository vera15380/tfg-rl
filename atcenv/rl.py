import collections
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque
import random
from typing import Tuple
import numpy as np
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from atcenv import wandb_graphs
from main import WANDB_USAGE


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience) -> None:
        """Add experience to the buffer.
        Args:
            experience: tuple (state, action, reward, next_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        """Returns random sample from the buffer.
        Args:
            batch_size: size of the sample"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
        )

    def pop(self):
        """Returns the last experience saved to the buffer.
        In this case only the states and actions are returned."""
        states, actions, rewards, next_states = self.buffer.pop()
        return np.array(states), np.array(actions)


class PrioritizedReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        importance = 1 / len(self.buffer) * 1 / probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size, priority_scale=1.0):
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=batch_size, weights=sample_probs)
        probabilities, states, actions, rewards, next_states = zip(*(self.buffer[idx] for idx in sample_indices))
        importance = self.get_importance(probabilities)
        return (
            np.array(probabilities),
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
        )

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset


class NeuralNetwork(nn.Module):
    def __init__(self, n_obs_individual, n_output, n_hidden):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.input = nn.Linear(n_obs_individual, n_hidden)
        self.input.weight.data.normal_(0, 0.1)          # initialization
        self.hidden = nn.Linear(n_hidden, n_hidden)
        self.hidden.weight.data.normal_(0, 0.1)         # initialization
        # output --> action number to output
        self.output = nn.Linear(n_hidden, n_output)
        self.output.weight.data.normal_(0, 0.1)         # initialization

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        x = F.relu(x)
        return x


class DQN:
    def __init__(self, max_memory_size, batch_size, gamma, tau, lr, exploration_max, exploration_min, exploration_decay,
                 env, replay_buffer, hidden_neurons, target_update, max_episodes, trained_net=None):
        """
        Initializes the network and parameters needed for DQN policy.

        Parameters:
            max_memory_size: prioritized replay buffer max length
            batch_size: number of random samples taken from replay buffer to train at once.
            gamma: long term reward (1) or short term reward focus (0) (range from 0 to 1)
            tau:
            lr: learning rate of the optimizer. How quickly the neural network updates the concepts it has learned.

        """
        self.obs_dim = env.observation_space[0].shape[0]
        self.action_size = env.action_space[0].n
        self.n_flights = env.num_flights
        self.n_steps = 0
        self.gamma = gamma
        self.tau = tau
        self.exploration_max = exploration_max
        self.exploration_min = exploration_min
        self.exploration_decay = exploration_decay
        self.exploration_rate = exploration_max
        self.max_episodes = max_episodes
        self.replay_buffer = replay_buffer
        self.memory_sample_size = batch_size
        self.target_update = target_update
        self.hidden_neurons = hidden_neurons
        self.policy_net = NeuralNetwork(self.obs_dim, self.action_size, hidden_neurons)
        self.target_net = NeuralNetwork(self.obs_dim, self.action_size, hidden_neurons)
        if trained_net is not None:
            self.policy_net.load_state_dict(torch.load(f'./target_net/{trained_net}'))
            self.target_net.load_state_dict(torch.load(f'./target_net/{trained_net}'))
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr)
        self.loss_func = torch.nn.SmoothL1Loss()

    def select_action(self, obs, episode):
        """Selects action with the epsilon greedy strategy.
        The actions are chosen between random decisions and the decision from the policy net depending on the
        probability of exploration"""
        self.exploration_rate = self.exploration_min + (self.exploration_max - self.exploration_min) * \
                                math.exp(-1. * self.n_steps * self.exploration_decay)

        if random.random() > self.exploration_rate:
            q_eval = self.policy_net.forward(torch.Tensor(obs))
            action = q_eval[0].max(0)[1].cpu().data.item()
        else:
            action = random.randint(0, self.action_size - 1)

        self.n_steps += 1
        return action

    def soft_update(self):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def learn(self):
        # for now, we are not taking advantage of the prioritized buffer.
        states, actions, rewards, next_states = self.replay_buffer.sample(self.memory_sample_size)

        # we convert the batches to torch tensors
        rew_s = torch.FloatTensor(rewards).unsqueeze(-1)                      # batch_size, 1
        obs_s = torch.FloatTensor(np.array(states))                           # batch_size, n_obs
        actions_s = torch.LongTensor(actions).unsqueeze(-1)                   # batch_size, 1
        next_obs_s = torch.FloatTensor(np.array(next_states))                 # batch_size, n_obs
        # Get expected Q values from local model
        q_policy = self.policy_net(obs_s).gather(1, actions_s)
        # Get max predicted Q values (for next states) from target model
        q_next = (self.target_net(next_obs_s).detach()).max(1)[0]
        # Compute Q targets for current states
        q_target = rew_s + self.gamma * q_next.view(self.memory_sample_size, 1)

        # Compute the loss
        loss = self.loss_func(q_policy, q_target)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        if self.n_steps % self.target_update == 0:
            self.soft_update()
        return loss

    def train_model(self, env, comparison_env, EPISODES, RENDERING_FREQUENCY, render, CLOSING):
        """Training the DQN model and representing results both in wandb and tensorboard."""
        Experience = collections.namedtuple('Experience', field_names=['states', 'actions', 'rewards', 'next_states'])
        short_memo = ReplayBuffer(2)
        writer = SummaryWriter(comment='TFG')
        successful_rate_list = []
        cumulative_reward = 0
        do_nothing = [0] * comparison_env.num_flights
        for episode in tqdm(range(EPISODES)):
            obs, state_env = env.reset()
            c_obs = comparison_env.comparison_reset(state_env)
            successful_rate = 0
            n_turns_episode = 0
            done = False
            rew_episode = 0
            while not done:
                loss = np.NaN
                actions = []

                # selecting actions for each flights
                for i in range(env.num_flights):
                    action = self.select_action(obs, episode)
                    actions.append(action)

                rew, next_obs, done = env.step(actions)
                c_rew, c_obs, c_done = comparison_env.step(do_nothing)

                for i in range(0, env.num_flights):
                    # adding experiences to buffer
                    if i not in env.done:
                        experience = Experience(obs[i], actions[i], rew[i], next_obs[i])
                        self.replay_buffer.append(experience)

                if len(self.replay_buffer.buffer) > self.memory_sample_size + 1:
                    # if buffer size is enough, learn.
                    loss = self.learn()

                if WANDB_USAGE:
                    n_turns_step = sum(env.flights[i].n_turns for i in range(env.num_flights))
                    n_turns_episode += n_turns_step
                    rew_average = wandb_graphs.wandb_per_step(writer, env, comparison_env, n_turns_step, rew, loss, self)
                    rew_episode += rew_average

                obs = next_obs

                if (episode % RENDERING_FREQUENCY == 0 or episode == EPISODES - 1) and render:
                    env.render()
                    time.sleep(0.05)

                if CLOSING:
                    for x in rew:
                        if abs(x) >= 1400:
                            done = True
                            c_done = True

            if WANDB_USAGE:
                cumulative_reward += rew_episode
                wandb_graphs.wandb_per_episode(writer, env, comparison_env, rew_episode, n_turns_episode,
                                               successful_rate_list,
                                               episode, cumulative_reward, CLOSING)
            env.close()
            comparison_env.close()

        successful_rate_perc = (sum(successful_rate_list) / len(successful_rate_list)) * 100
        print(f"Training done. The success rate was of: {successful_rate_perc} %")

        if WANDB_USAGE:
            wandb.config.update({"success_rate of episodes": successful_rate_perc})

        writer.close()
        return successful_rate_perc



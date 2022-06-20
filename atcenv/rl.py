
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import deque
import random
from typing import Tuple


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience) -> None:
        """Add experience to the buffer.

        Args:
            experience: tuple (state, action, reward, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states = zip(*(self.buffer[idx] for idx in indices))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
        )

    def pop(self):
        states, actions, rewards, next_states = self.buffer.pop()
        return np.array(states), np.array(actions)


class NeuralNetwork(nn.Module):
    def __init__(self, n_obs_individual, n_output, n_hidden):
        super().__init__()

        # Inputs to hidden layer linear transformation
        self.input = nn.Linear(n_obs_individual, n_hidden)
        self.hidden = nn.Linear(n_hidden, n_hidden)
        # output --> action number to output
        self.output = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # Pass the input tensor through each of our operations
        x = self.input(x)
        x = self.hidden(x)
        x = self.output(x)
        x = F.relu(x)
        return x


class DQN:
    def __init__(self, max_memory_size, batch_size, gamma, tau, lr, exploration_max, exploration_min, exploration_decay, env, replay_buffer, hidden_neurons,
                 target_update, max_episodes, trained_net=None):
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
        self.policy_net = NeuralNetwork(self.obs_dim, self.action_size, hidden_neurons)
        self.target_net = NeuralNetwork(self.obs_dim, self.action_size, hidden_neurons)
        if trained_net is not None:
            self.policy_net.load_state_dict(torch.load(f'./target_net/{trained_net}'))
            self.target_net.load_state_dict(torch.load(f'./target_net/{trained_net}'))
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr)
        self.loss_func = torch.nn.SmoothL1Loss()

    def select_action(self, obs, episode):
        # epsilon greedy strategy
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
            target_param.data.copy_(self.tau * policy_param.data + (1.0-self.tau) * target_param.data)

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

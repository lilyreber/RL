import os
import csv
from collections import deque
import numpy as np
from torch import nn
import torch


def create_network(input_dim, hidden_dims, output_dim):
    layers = []
    last_output_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(in_features=last_output_dim, out_features=hidden_dim))
        layers.append(nn.ReLU())
        last_output_dim = hidden_dim
    layers.append(nn.Linear(in_features=last_output_dim, out_features=output_dim))
    network = nn.Sequential(*layers)
    return network


def select_action_eps_greedy(Q, state, epsilon):
    if not isinstance(state, torch.Tensor):
        state = torch.tensor(state, dtype=torch.float32)
    Q_s = Q(state).detach().numpy()

    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(0, len(Q_s))
    else:
        action = np.argmax(Q_s)

    action = int(action)
    return action


def to_tensor(x, dtype=np.float32):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=dtype)
    x = torch.from_numpy(x)
    return x


def compute_td_target(Q, rewards, next_states, terminateds, gamma=0.99, check_shapes=True):
    r = to_tensor(rewards)
    s_next = to_tensor(next_states)
    term = to_tensor(terminateds, bool)

    with torch.no_grad():
        Q_sn = Q(s_next)
        V_sn = torch.max(Q_sn, dim=1)[0]
        V_sn[term] = 0

    assert V_sn.dtype == torch.float32

    target = r + gamma * V_sn

    if check_shapes:
        assert Q_sn.data.dim() == 2
        assert V_sn.data.dim() == 1
        assert target.data.dim() == 1

    return target


def compute_td_loss(Q, states, actions, td_target, regularizer=.1, out_non_reduced_losses=False):
    s = to_tensor(states)
    a = to_tensor(actions, int).long()

    Q_s_a = torch.gather(Q(s), dim=1, index=torch.unsqueeze(a, 1)).squeeze(-1)

    td_error = Q_s_a - td_target
    td_losses = td_error ** 2
    loss = torch.mean(td_losses)
    loss += regularizer * torch.abs(Q_s_a).mean()

    if out_non_reduced_losses:
        return loss, td_losses.detach()

    return loss


def linear(st, end, duration, t):
    if t >= duration:
        return end
    return st + (end - st) * (t / duration)


def sample_batch(replay_buffer, n_samples):
    indices = np.random.choice(len(replay_buffer), n_samples, replace=True)
    states, actions, rewards, next_states, terminateds = zip(*[replay_buffer[i] for i in indices])
    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(terminateds)

def symlog(x):
    return np.sign(x) * np.log(np.abs(x) + 1)

def softmax(xs, temp=1.):
    exp_xs = np.exp((xs - xs.max()) / temp)
    return exp_xs / exp_xs.sum()

def sample_prioritized_batch(replay_buffer, n_samples):
    priorities = [sample[0] for sample in replay_buffer]
    indices = np.random.default_rng(seed=42).choice(len(replay_buffer), n_samples, replace=True, p=softmax(symlog(priorities)))

    _, states, actions, rewards, next_states, terminateds = zip(
        *[replay_buffer[i] for i in indices]
    )

    batch = (
        np.array(states), np.array(actions), np.array(rewards),
        np.array(next_states), np.array(terminateds)
    )
    return batch, indices

def update_batch(replay_buffer, indices, batch, new_priority):
    states, actions, rewards, next_states, terminateds = batch

    for i in range(len(indices)):
        new_batch = (
            new_priority[i], states[i], actions[i], rewards[i],
            next_states[i], terminateds[i]
        )
        replay_buffer[indices[i]] = new_batch

def sort_replay_buffer(replay_buffer):
    new_rb = deque(maxlen=replay_buffer.maxlen)
    new_rb.extend(sorted(replay_buffer, key=lambda sample: sample[0]))
    return new_rb

class MyDQN:
    def __init__(self, env, hidden_dims=(64, 64), lr=1e-3, gamma=0.99,
                 eps_st=1.0, eps_end=0.01, eps_dur=0.1,
                 replay_buffer_size=50000, batch_size=32,
                 target_update_interval=500, train_freq=1,
                 sort_buffer_freq=40,
                 verbose=1):

        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.train_freq = train_freq
        self.sort_buffer_freq = sort_buffer_freq
        self.verbose = verbose

        self.eps_st = eps_st
        self.eps_end = eps_end
        self.eps_dur = eps_dur

        self.Q = create_network(self.state_dim, hidden_dims, self.action_dim)
        self.target_Q = create_network(self.state_dim, hidden_dims, self.action_dim)
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.target_Q.eval()

        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=lr)
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.episode_count = 0
        self.step_count = 0


    def learn(self, total_timesteps):
        s, _ = self.env.reset()
        episode_reward = 0
        episode_length = 0
        episode_loss = 0
        loss_count = 0

        for step in range(total_timesteps):
            epsilon = linear(self.eps_st, self.eps_end, self.eps_dur * total_timesteps, step)
            a = select_action_eps_greedy(self.Q, s, epsilon)

            s_next, r, terminated, truncated, _ = self.env.step(a)
            done = terminated or truncated

            with torch.no_grad():
                td_target = compute_td_target(
                    self.target_Q,
                    np.array([r]),
                    np.array([s_next]),
                    np.array([terminated]),
                    gamma=self.gamma
                )
                loss = compute_td_loss(
                    self.Q,
                    np.array([s]),
                    np.array([a]),
                    td_target
                )
                loss = loss.item()

            self.replay_buffer.append((loss, s, a, r, s_next, terminated))

            s = s_next
            episode_reward += r
            episode_length += 1
            self.step_count += 1

            if step % self.train_freq == 0 and len(self.replay_buffer) >= self.batch_size:
                train_batch, indices = sample_prioritized_batch(self.replay_buffer, self.batch_size)
                states, actions, rewards, next_states, terminateds = train_batch

                self.optimizer.zero_grad()
                td_target = compute_td_target(self.target_Q, rewards, next_states, terminateds, gamma=self.gamma)
                loss, td_losses = compute_td_loss(self.Q, states, actions, td_target, out_non_reduced_losses=True)
                loss.backward()
                self.optimizer.step()

                update_batch(self.replay_buffer, indices, train_batch, td_losses.numpy())

                loss = loss.item()

                if loss > 0:
                    episode_loss += loss
                    loss_count += 1

            if step % self.sort_buffer_freq == 0:
                self.replay_buffer = sort_replay_buffer(self.replay_buffer)

            if step % self.target_update_interval == 0:
                self.target_Q.load_state_dict(self.Q.state_dict())

            if done:
                self.episode_count += 1
                avg_loss = episode_loss / loss_count if loss_count > 0 else 0

                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                self.losses.append(avg_loss)


                s, _ = self.env.reset()
                episode_reward = 0
                episode_length = 0
                episode_loss = 0
                loss_count = 0



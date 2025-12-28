"""Q-learning Agent class."""
import numpy as np

from sumo_rl.exploration.epsilon_greedy import EpsilonGreedy


class MyQLAgent:
    """Q-learning Agent class."""

    def __init__(self, starting_state, state_space, action_space, alpha=0.5, gamma=0.95, exploration_strategy=EpsilonGreedy()):
        """Initialize Q-learning agent."""
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.action = None
        self.alpha = alpha
        self.gamma = gamma
        self.q_table = {self.state: [0 for _ in range(action_space.n)]}
        self.exploration = exploration_strategy
        self.acc_reward = 0

    def act(self):
        """Choose action based on Q-table."""
        self.action = self.exploration.choose(self.q_table, self.state, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False, algorithm='q'):
        """Update Q-table with new experience."""
        if next_state not in self.q_table:
            self.q_table[next_state] = [0 for _ in range(self.action_space.n)]

        s = self.state
        s1 = next_state
        a = self.action

        td_error = 0

        if done:
            td_error = reward - self.q_table[s][a]
        else:
            if algorithm == 'q':
                V_ns = np.max(self.q_table[s1])
                td_error = reward + self.gamma * V_ns - self.q_table[s][a]
            if algorithm == "sarsa":
                decay = self.exploration.decay
                self.exploration.decay = 1
                a1 = self.exploration.choose(self.q_table, next_state, self.action_space)
                self.exploration.decay = decay
                td_error = reward + self.gamma * self.q_table[s1][a1] - self.q_table[s][a]


        self.q_table[s][a] += self.alpha * td_error
        self.state = s1
        self.acc_reward += reward
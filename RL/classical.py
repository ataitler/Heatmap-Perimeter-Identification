import random
from RL.DQN import Logger
from itertools import combinations
import numpy as np
import pickle
import os

class TablePolicy(object):
    def __init__(self, actions, lr=0.1, gamma=0.99):
        self.table = {}
        self.gamma = gamma
        self.num_actions = actions
        self.actions = np.zeros(actions)
        self.alpha = lr

    def get_action(self, state):
        hstate = str(state)
        if hstate in self.table:
            return self.table[hstate].argmax()
        else:
            return random.randrange(self.num_actions)

    def get_Q_values(self, state):
        hstate = str(state)
        if hstate in self.table:
            return self.table[hstate]
        else:
            return self.actions.copy()

    def update(self, state, action, reward, next_state):
        hstate = str(state)
        if hstate not in self.table:
            self.table[hstate] = self.actions.copy()
        self.table[hstate][action] = (1-self.alpha) * self.table[hstate][action] + self.alpha * (reward + self.gamma * self.get_Q_values(next_state).max())

    def render(self):
        mat = []
        for state in self.table.keys():
            mat.append(self.table[state])
        return mat

    def __len__(self):
        return len(self.table.keys())

    def save_policy(self, file_name=None):
        if file_name is not None:
            with open(file_name, 'wb') as f:
                pickle.dump(self.table, f)

    def load_policy(self, file_name=None):
        if file_name is not None:
            if os.path.exists(file_name):
                with open(file_name, 'rb') as f:
                    self.table = pickle.load(f)

class QLearningAgent(object):
    def __init__(self, actions, eps = 0.1, lr=0.1, gamma=0.99, log_name=None):
        self.gamma = gamma
        self.actions = actions
        self.eps = eps
        # self.states = states
        self.policy = TablePolicy(actions, lr)
        self.Logger = Logger(file_name=log_name)
        self.log_episode = 1

    def sample_action(self, state, force_explore=False, exploit=False):
        if exploit:
            return self.policy.get_action(state)
        explore = random.random()
        if explore <= self.eps or force_explore:
            action = random.randrange(self.actions)
        else:
            action = self.policy.get_action(state)
        return action

    def update_policy(self, state, action, reward, next_state):
        self.policy.update(state, action, reward, next_state)

    def _init_table(self):
        pass

    def save_policy(self, file_name):
        policy_file = os.path.join('models', file_name)
        self.policy.save_policy(policy_file)

    def load_policy(self, file_name):
        policy_file = os.path.join('models', file_name)
        self.policy.load_policy(policy_file)

    def log(self, actions, rewards):
        self.Logger.log_episode(episode=self.log_episode, actions=actions, rewards=rewards, model=None)
        self.log_episode += 1

    def get_len(self):
        return len(self.policy)

    def render_state(self):
        return self.policy.render()



from collections import namedtuple, deque
import random

import torch

from RL.networks import Model
import torch.optim as optim


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQNAgent(object):
    def __init__(self, actions, batch_size=128, gamma=0.99, eps=0.1, lr=1e-4, tau=0.005, memory=10000, clip=100):
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps = eps
        self.lr = lr
        self.tau = tau
        self.num_actions = actions
        self.clip = clip

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = Model(out=actions)
        self.target_net = Model(out=actions)
        if self.device != "cpu":
            self.policy_net.cuda()
            self.target_net.cuda()
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.lr)

        self.memory = ReplayMemory(memory)


    def sample_action(self, state, force_explore=False):
        explore = random.random()
        if explore <= self.eps or force_explore:
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.int64)
        else:
            with torch.no_grad():
                if self.device != "cpu":
                    return self.policy_net(state).max(1).indices.view(1, 1).cuda()
                else:
                    return self.policy_net(state).max(1).indices.view(1, 1)

    def store_transition(self, state, action, reward, next_state):
        self.memory.push(state, action, reward, next_state)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # sample batch of random transitions
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a)
        print("states tensor", state_batch.get_device())
        print("actions tensor", action_batch.get_device())
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip)
        self.optimizer.step()

        # update the target network
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

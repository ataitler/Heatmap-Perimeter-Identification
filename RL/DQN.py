from collections import namedtuple, deque
import random
import os

import torch

from RL.networks import Simple, LeNet5
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class Logger(object):
    def __init__(self, file_name=None, tb_name=None):
        self.logs_dir = "logs"
        self.tb_dir = "runs"
        self.file_name = None
        if file_name is not None:
            self.file_name = os.path.join(self.logs_dir, file_name)
            fh = open(self.file_name, 'w+')
            fh.close()
        self.tb_name = None
        if tb_name is not None:
            self.tb_name = os.path.join(self.tb_dir, tb_name)
            self.TBwriter = SummaryWriter(self.tb_name)

    def log_episode(self, episode, actions, rewards, model):
        if self.file_name is not None:
            msg = ("######################\n# Episode " + str(episode) + " \n######################\n")
            msg += "step,action,reward\n"
            for i in range(len(actions)):
                msg += str(i) + "," + str(actions[i]) + "," + str(rewards[i]) + "\n"

            fh = open(self.file_name, 'a')
            fh.write(msg)
            fh.close()

        if self.tb_name is not None:
            total_reward = sum(rewards)
            self.TBwriter.add_scalar('Reward', total_reward, episode)
            conv_idx = 1
            linear_idx = 1
            for mod in model.modules():
                if isinstance(mod, nn.Conv2d):
                    weights = mod.weight
                    weights_shape = weights.shape
                    num_kernels = weights_shape[0]
                    for k in range(num_kernels):
                        flattened_weights = weights[k].flatten()
                        tag = f"Conv2d_{conv_idx}/kernel_{k}"
                        self.TBwriter.add_histogram(tag, flattened_weights, global_step=episode)
                    conv_idx += 1
                if isinstance(mod, nn.Linear):
                    weights = mod.weight
                    biases = mod.bias
                    flattened_weights = weights.flatten()
                    flattened_biases = biases.flatten()
                    tag = f"Linear{linear_idx}"
                    self.TBwriter.add_histogram(tag+'/weights', flattened_weights, global_step=episode)
                    self.TBwriter.add_histogram(tag+'/bias', flattened_biases, global_step=episode)
                    linear_idx += 1


class ReplayMemory(object):

    def __init__(self, capacity):
        self.max_capacity = capacity
        self.memory = deque([], maxlen=capacity)
        self.allocated = 0
        # self.head = 0

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        if len(self.memory) <= self.max_capacity:
            for e in args:
                self.allocated += e.element_size() * e.nelement()

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def size_of(self):
        return self.allocated


class DQNAgent(object):
    def __init__(self, actions, batch_size=32, gamma=0.99, eps=0.1, lr=1e-4, tau=0.005, memory=10000, clip=100):
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps = eps
        self.lr = lr
        self.tau = tau
        self.num_actions = actions
        self.clip = clip
        # self.log = False
        self.log_episode = 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # self.policy_net = Model(out=actions)
        # self.target_net = Model(out=actions)
        self.policy_net = LeNet5(out=actions)
        self.target_net = LeNet5(out=actions)
        if torch.cuda.is_available():
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
            if torch.cuda.is_available():
                state = state.cuda()
            with torch.no_grad():
                action = self.policy_net(state).max(1).indices.view(1, 1)
                if torch.cuda.is_available():
                    return action.cpu()
                return action

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
        # non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
        #                                         batch.next_state)), device=self.device, dtype=torch.bool)
        # non_final_next_states = torch.cat([s for s in batch.next_state
        #                                    if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_batch = torch.cat(batch.next_state)

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_batch = next_batch.cuda()

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        # Compute V(s_{t+1}) for all next states.
        # next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            # next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
            next_state_values = self.target_net(next_batch).max(1).values

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
            target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                        1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def get_buffer_size(self):
        return self.memory.size_of()

    def get_buffer_len(self):
        return len(self.memory)

    def set_logger(self, logs_name=None, tb_name=None):
        # self.log = True
        self.Logger = Logger(file_name=logs_name, tb_name=tb_name)

    def log(self, actions, rewards):
        # log file + Tensorboard
        self.Logger.log_episode(episode=self.log_episode, actions=actions, rewards=rewards, model=self.policy_net)
        self.log_episode += 1

    def save_agent_state(self, checkpoint):
        torch.save({
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, 'models/'+checkpoint)

    def load_agent_state(self, checkpoint):
        if os.path.exists('models/'+checkpoint):
            checkpoint = torch.load('models/'+checkpoint)
            self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
import sys
import numpy as np
import argparse
from env import PIEnv
from RL.DQN import DQNAgent
from RL.networks import SimpleMLP
import torch
# import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=15)
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--updates', type=int, default=1)
parser.add_argument('--explore', type=int, default=128)
parser.add_argument('--batch', type=int, default=32)
parser.add_argument('--verbose', type=bool, action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--buffer', type=int, default=10000)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--log', type=str, default=None)
parser.add_argument('--log_ratio', type=int, default=1)
parser.add_argument('--slim_log', type=bool, action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--tb', type=str, default=None)
parser.add_argument('--reg', type=float, default=0.000001)
args = parser.parse_args()

def main():
    pure_exploration_steps = args.explore
    n_update_steps = args.updates
    num_episodes = args.episodes
    num_steps = args.steps
    b2M = 1024*1024
    # print(args.log_ratio)
    # sys.exit()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # transform = transforms.Compose([transforms.ToTensor()])

    env = PIEnv(map="data/rsz_heat_map_with_green2.jpg", clean="data/rsz_heat_map.jpg", regularizer=args.reg)
    Agent = DQNAgent(actions=env.action_space.n, batch_size=args.batch, memory=args.buffer, lr=args.lr)
    Agent.set_logger(logs_name=args.log, tb_name=args.tb, slim_log=args.slim_log)
    # Agent.set_network(SimpleMLP, env.heat_map)

    if args.verbose:
        print('Running on device:', device)
        print('Running for', args.episodes, 'episodes with', args.steps, 'steps each. Batch size:', args.batch)
        print('GPU usage at start:', torch.cuda.memory_allocated()/b2M, "MB")
        print('RB size at start:', Agent.get_buffer_size() / b2M, "MB")

    for i_episode in range(num_episodes):
        # state, _ = env.reset(seed=20)
        state, _ = env.reset(seed=29)
        # state = torch.tensor(state, device=device).unsqueeze(0)
        state = torch.tensor(state).unsqueeze(0)
        actions = []
        rewards = []
        total_reward = 0
        for step in range(num_steps):
            action = Agent.sample_action(state, force_explore=(i_episode*num_steps < pure_exploration_steps))

            # apply the action
            obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward = total_reward + reward
            actions.append(action.item())
            rewards.append(reward)
            # reward = torch.tensor([reward], device=device)
            reward = torch.tensor([reward])
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                # next_state = torch.tensor(obs, device=device).unsqueeze(0)
                next_state = torch.tensor(obs).unsqueeze(0)
                # next_state = transform(obs).unsqueeze(0)
                # if torch.cuda.is_available():
                #     next_state = next_state.cuda()

            # Store the transition in memory
            Agent.store_transition(state, action, reward, next_state)

            # Move to the next state
            state = next_state

            # If the episode is up, then start another one
            if done:
                env.reset()

        print('Episode', i_episode+1, 'ended with reward:', total_reward)
        if args.verbose:
            print('GPU usage after', i_episode, 'episodes:', torch.cuda.memory_allocated()/b2M, "MB")
            print('RB size after', i_episode, 'episodes:', Agent.get_buffer_size() / b2M, "MB with", Agent.get_buffer_len(), "elements")
        for update in range(n_update_steps):
            # Perform one step of the optimization (on the policy network)
            if args.verbose:
                print('\tUpdate step', update+1, '...... ', end='')
            Agent.optimize_model()
            if args.verbose:
                print('Done')
        if (i_episode) % args.log_ratio == 0:
            Agent.log(actions=actions, rewards=rewards, ratio=args.log_ratio)

    # Close the env
    env.close()

    #plot the reward graph








if __name__ == "__main__":
    main()


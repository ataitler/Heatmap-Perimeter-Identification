import sys
import numpy as np
import argparse
from env import PIEnv
from RL.DQN import DQNAgent
import torch
import torchvision.transforms as transforms

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=100)
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--updates', type=int, default=5)
parser.add_argument('--explore', type=int, default=128)
args = parser.parse_args()

def main():
    pure_exploration_steps = args.explore
    n_update_steps = args.updates
    num_episodes = args.episodes
    num_steps = args.steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Running on device:', device)
    transform = transforms.Compose([transforms.ToTensor()])
    env = PIEnv(map="data/rsz_heat_map_with_green.jpg", clean="data/rsz_heat_map.jpg")

    Agent = DQNAgent(env.action_space.n)

    for i_episode in range(num_episodes):
        state, _ = env.reset(seed=20)
        state = transform(state).unsqueeze(0)
        if torch.cuda.is_available():
            state = state.cuda()

        total_reward = 0
        for step in range(num_steps):
            action = Agent.sample_action(state, force_explore=(i_episode*num_steps < pure_exploration_steps))
            # print((state.element_size()*state.nelement())/1024/1024, state.shape)
            # print(sys.getsizeof(action))
            # sys.exit()
            # apply the action
            obs, reward, terminated, truncated, _ = env.step(action.item())
            total_reward = total_reward + reward
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = transform(obs).unsqueeze(0)
                if torch.cuda.is_available():
                    next_state = next_state.cuda()

            # Store the transition in memory
            Agent.store_transition(state, action, reward, next_state)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            Agent.optimize_model()

            # If the episode is up, then start another one
            if done:
                env.reset()

        print('Episode', i_episode , 'ended with reward:', total_reward)
        for update in range(n_update_steps):
            # Perform one step of the optimization (on the policy network)
            print('\tUpdate step', update+1, '...... ', end='')
            Agent.optimize_model()
            print('Done')

    # Close the env
    env.close()

    #plot the reward graph








if __name__ == "__main__":
    main()


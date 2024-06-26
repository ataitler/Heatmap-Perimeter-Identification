import argparse
from env import PIEnv
from RL.classical import QLearningAgent
import time

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=50000, help="how many episode to run")
parser.add_argument('--steps', type=int, default=4, help="how many steps per episode")
parser.add_argument('--updates', type=int, default=5, help="how many gradient updates at the end of each episode")
parser.add_argument('--explore', type=int, default=128, help="how many random exploration steps at the beginning of the training")
parser.add_argument('--batch', type=int, default=32, help="number of states in traning batch")
parser.add_argument('--verbose', type=bool, action=argparse.BooleanOptionalAction, default=False, help="verobse mode")
parser.add_argument('--buffer', type=int, default=10000, help="samples capacity in the replay buffer")
parser.add_argument('--lr', type=float, default=0.0001, help="learning rate")
parser.add_argument('--log', type=str, default=None, help="log each episode actions and rewards per action")
parser.add_argument('--log_ratio', type=int, default=10, help="the episode log ratio")
parser.add_argument('--slim_log', type=bool, action=argparse.BooleanOptionalAction, default=True, help="only reward logging in tensorboard (or weights also)")
parser.add_argument('--tb', type=str, default=None, help="tensorboard output file")
parser.add_argument('--reg', type=float, default=1, help="reward regularaizer (unused currently)")
parser.add_argument('--network', type=str, default="LeNet5", help="network type, can choose between LeNet5, SimpleMLP and SimpleCNN")
parser.add_argument('--reduce', type=float, default=1.0, help="images size down-sampling ratio")
parser.add_argument('--policy', type=str, default=None, help="policy file")
args = parser.parse_args()

def main():
    pure_exploration_steps = args.explore
    num_episodes = args.episodes
    num_steps = args.steps

    env = PIEnv(map="data/rsz_heat_map_with_green2.jpg", clean="data/rsz_heat_map.jpg", regularizer=args.reg, reduce=args.reduce)
    Agent = QLearningAgent(actions=env.action_space.n, lr=args.lr)
    Agent.load_policy(args.policy)


    if args.verbose:
        print('Running for', args.episodes, 'episodes with', args.steps, 'steps each. Batch size:', args.batch)

    start_time = time.time()
    all_rewards = []
    for i_episode in range(num_episodes):
        _, _ = env.reset(seed=29)
        state = env.get_convex_set()

        actions = []
        rewards = []
        total_reward = 0
        for step in range(num_steps):
            action = Agent.sample_action(state, force_explore=(i_episode*num_steps < pure_exploration_steps))

            # apply the action
            _, reward, terminated, truncated, _ = env.step(action)

            obs = env.get_convex_set()
            total_reward = total_reward + reward
            actions.append(action)
            rewards.append(reward)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = obs

            # Store the transition in memory
            Agent.update_policy(state, action, reward, next_state)

            # Move to the next state
            state = next_state

            # If the episode is up, then start another one
            if done:
                env.reset()

        all_rewards.append(total_reward)
        if args.verbose:
            if (i_episode) % args.log_ratio == 0:
                # all_rewards.append(total_reward)
                print('Episode', i_episode + 1, 'ended with reward:', total_reward, "table len:", Agent.get_len())
        Agent.log(actions=actions, rewards=rewards)

    Agent.save_policy(args.policy)
    # Close the env
    env.close()

    if args.verbose:
        print("Execution time:", time.time()-start_time)
        # plot the reward graph
        window = 50
        averaged = all_rewards[:int(window/2)]
        for i in range(int(window/2),len(all_rewards)-int(window/2)):
            averaged.append(sum(all_rewards[i - int(window/2):i + int(window/2)])/window)
        # plt.plot(all_rewards, 'blue', averaged,'red')
        plt.plot(averaged)
        plt.show()


    #plot the reward graph








if __name__ == "__main__":
    main()


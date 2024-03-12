import sys
import cv2
import numpy as np
from env import PIEnv
from RL.DQN import DQNAgent
import torch
import torchvision.transforms as transforms

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([transforms.ToTensor()])
    env = PIEnv(map="data/heat_map_with_green.jpg", clean="data/heat_map.jpg")
    num_actions = env.action_space.n

    Agent = DQNAgent(env.action_space.n)

    num_episodes = 10
    num_steps = 3
    for i_episode in range(num_episodes):
        print('##############################################')
        print('############ Episode', i_episode+1, 'starting #############')
        print('##############################################')
        state, _ = env.reset(seed=42)
        env.render()
        sys.exit()
        state = transform(state).unsqueeze(0)
        for step in range(num_steps):
            action = Agent.sample_action(state, force_explore=(i_episode*num_steps < 20))

            # apply the action
            obs, reward, terminated, truncated, _ = env.step(action.item())
            print(action.item(), reward)
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = transform(obs).unsqueeze(0)

            # Store the transition in memory
            Agent.store_transition(state, action, reward, next_state)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            Agent.optimize_model()

            # If the episode is up, then start another one
            if done:
                env.reset()

        # Close the env
        # env.render()
        env.close()



    # cv2.imshow('Image with Convex Hull Around Perimeter Area', state)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()








if __name__ == "__main__":
    main()


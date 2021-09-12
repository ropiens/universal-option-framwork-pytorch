import os

import gym
import numpy as np
import torch

import asset
from agent.UOF import UOF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test():

    #################### Hyperparameters ####################
    env_name = "MountainCarContinuous-h-v1"  # "MountainCarContinuous-v0"
    max_episodes = 5  # max num of training episodes
    random_seed = 0
    render = True

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    """
     Actions (both primitive and subgoal) are implemented as follows:
       action = ( network output (Tanh) * bounds ) + offset
       clip_high and clip_low bound the exploration noise
    """

    # primitive action bounds and offset
    action_bounds = env.action_space.high[0]
    action_offset = np.array([0.0])
    action_offset = torch.FloatTensor(action_offset.reshape(1, -1)).to(device)
    action_clip_low = np.array([-1.0 * action_bounds])
    action_clip_high = np.array([action_bounds])

    # state bounds and offset
    state_bounds_np = np.array([0.9, 0.07])
    state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1)).to(device)
    state_offset = np.array([-0.3, 0.0])
    state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)
    state_clip_low = np.array([-1.2, -0.07])
    state_clip_high = np.array([0.6, 0.07])

    # exploration noise std for primitive action and subgoals
    exploration_action_noise = np.array([0.1])
    exploration_state_noise = np.array([0.02, 0.01])

    goal_state = np.array([0.48, 0.04])  # final goal state to be achived
    threshold = np.array([0.01, 0.02])  # threshold value to check if goal state is achieved

    # DDPG parameters:
    gamma = 0.95  # discount factor for future rewards
    lr = 0.001

    # save trained models
    directory = "{}/preTrained/{}/".format(os.getcwd(), env_name)
    filename = "UOF_{}".format(env_name)
    #########################################################

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # creating UOF agent and setting parameters
    agent = UOF(
        state_dim,
        action_dim,
        render,
        threshold,
        action_bounds,
        action_offset,
        state_bounds,
        state_offset,
        lr,
        gamma,
    )

    # load agent
    agent.load(directory, filename)

    # Evaluation
    for i_episode in range(1, max_episodes + 1):

        agent.reward = 0
        agent.timestep = 0

        state = env.reset()
        agent.run_UOF(env, state, goal_state)

        print("Episode: {}\t Reward: {}\t len: {}".format(i_episode, agent.reward, agent.timestep))

    env.close()


if __name__ == "__main__":
    test()

import os

import gym
import numpy as np
import torch

import asset
from agent.UOF import UOF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train():
    #################### Hyperparameters ####################
    env_name = "MountainCarContinuous-h-v1"
    save_episode = 10  # keep saving every n episodes
    max_episodes = 1000  # max num of training episodes
    random_seed = 0
    render = True

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    option_dim = 8
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
    # state bounds and offset
    state_bounds_np = np.array([0.9, 0.07])
    state_bounds = torch.FloatTensor(state_bounds_np.reshape(1, -1)).to(device)
    state_offset = np.array([-0.3, 0.0])
    state_offset = torch.FloatTensor(state_offset.reshape(1, -1)).to(device)

    goal_state = np.array([0.48, 0.04])  # final goal state to be achived
    threshold = np.array([0.05, 1.0])  # threshold value to check if goal state is achieved
    # (not considering velocity)

    # DDPG & DIOL parameters:
    gamma = 0.95  # discount factor for future rewards
    tau = 0.15  # target soft update rate
    n_iter = 100  # update policy n_iter times in one DDPG update
    batch_size = 100  # num of transitions sampled from replay buffer
    lr = 0.001

    # time horizon to step with a subgoal
    H = 100
    testing_peoriod = 10

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
        env,
        state_dim,
        option_dim,
        action_dim,
        render,
        threshold,
        action_bounds,
        action_offset,
        lr,
        gamma,
        tau,
        H,
        use_aaes=True,
    )

    # logging file:
    log_f = open("log.txt", "w+")

    # training procedure
    for i_episode in range(1, max_episodes + 1):
        agent.reward = 0
        agent.timestep = 0

        state = env.reset(test=False)
        # collecting experience in environment
        last_state, done = agent.run_UOF(state, goal_state, option_dim)

        # update all levels
        agent.update(n_iter, batch_size)
        agent.training_ep_count += 1

        if agent.training_ep_count % testing_peoriod == 0:
            state = env.reset(test=True)
            # collecting experience in environment
            last_state, done = agent.run_UOF(state, goal_state, option_dim)

            if agent.actor.use_aaes:
                agent.actor.actor_exploration.update_success_rates()

        # logging updates:
        log_f.write("{},{}\n".format(i_episode, agent.reward))
        log_f.flush()

        if agent.check_goal(last_state, goal_state, threshold):
            print("################ Solved! ################ ")
            name = filename + "_solved"
            agent.save(directory, name)

        if i_episode % save_episode == 0:
            agent.save(directory, filename)

        print("Episode: {}\t Reward: {}".format(i_episode, agent.reward))


if __name__ == "__main__":
    train()

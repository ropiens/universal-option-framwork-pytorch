import numpy as np
import torch
import time
from collections import namedtuple

from .utils import LowLevelHindsightReplayBuffer, HighLevelHindsightReplayBuffer

from .DDPG import DDPG
from .DIOL import DIOL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


OPT_Tr = namedtuple(
    "transition",
    (
        "state",
        "desired_goal",
        "option",
        "next_state",
        "achieved_goal",
        "option_done",
        "reward",
        "done",
    ),
)
ACT_Tr = namedtuple(
    "transition",
    (
        "state",
        "desired_goal",
        "action",
        "next_state",
        "achieved_goal",
        "reward",
        "done",
    ),
)


class UOF:
    def __init__(
        self,
        state_dim,
        option_dim,
        action_dim,
        render,
        threshold,
        action_bounds,
        action_offset,
        state_bounds,
        state_offset,
        lr,
        gamma,
    ):

        """Inter-Option/High-Level policies - DIOL"""
        # opt, optor refer to high-level policy
        optor_mem_capacity = int(1e5)
        self.optor = DIOL(state_dim, option_dim, lr, gamma)
        self.optor_replay_buffer = HighLevelHindsightReplayBuffer(optor_mem_capacity, OPT_Tr)

        """Intra-Option/Low-Level policies - DDPG + HER"""
        # act, actor refer to low-level policy
        actor_mem_capacity = int(1e5)
        self.actor = DDPG(state_dim, action_dim, action_bounds, action_offset, lr, gamma)
        self.actor_replay_buffer = LowLevelHindsightReplayBuffer(actor_mem_capacity, ACT_Tr)

        # set some parameters
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.threshold = threshold
        self.render = render

        # logging parameters
        self.goals = [None]
        self.reward = 0
        self.timestep = 0

    def check_goal(self, state, goal, threshold):
        for i in range(self.state_dim):
            if abs(state[i] - goal[i]) > threshold[i]:
                return False
        return True

    def run_UOF(self, env, state, goal):
        next_state = None
        done = False
        new_episode = True
        # logging updates
        self.goals[0] = goal

        while not done:
            goal_achieved = False
            new_option = True
            option = self.optor.select_option(state, goal)
            while (not done) and (not goal_achieved):
                if self.render:
                    env.render()

                #   <================ low level policy ================>
                # take primitive action
                action = self.actor.select_action(state, goal, final_goal_ind)

                next_state, rew, done, _ = env.step(action)

                # this is for logging
                self.reward += rew
                self.timestep += 1

                # check if goal is achieved
                goal_achieved = self.check_goal(next_state, goal, self.threshold)
                reward = 0.0 if goal_achieved else -1.0
                # ('state', 'desired_goal', 'action', 'next_state', 'achieved_goal', 'reward', 'done')
                self.actor_replay_buffer.store_experience(
                    new_episode,
                    state,
                    goal,
                    action,
                    next_state,
                    next_state,
                    reward,
                    float(done),
                )

                state = next_state
                new_episode = False

        return next_state, done

    def update(self, n_iter, batch_size):
        self.actor.update(self.actor_replay_buffer, n_iter, batch_size)

    def save(self, directory, name):
        self.actor.save(directory, name + "_level_{}".format(0))

    def load(self, directory, name):
        self.actor.load(directory, name + "_level_{}".format(0))

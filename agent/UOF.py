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

    def set_subgoal(self, option_ind):
        goals = np.array([[0.15, 0.04], [0.00, 0.04], [-0.68, 0.04], [-1.1, 0.04]])
        return goals[option_ind]

    def run_UOF(self, env, state, goal):
        next_state = None
        time_done = False
        new_episode = True
        # logging updates
        self.goals[0] = goal

        while not time_done:
            subgoal_done = False
            new_option = True

            # get subgoal from optor
            option = self.optor.select_option(state, goal)
            subgoal = self.set_subgoal(option)
            while (not time_done) and (not subgoal_done):
                if self.render:
                    env.unwrapped.render_goal(goal, subgoal)

                # take action from actor
                action = self.actor.select_action(state, subgoal)
                next_state, act_reward, time_done, _ = env.step(action)

                # this is for logging
                self.reward += act_reward
                self.timestep += 1

                # check if goal is achieved
                subgoal_done = self.check_goal(next_state, subgoal, self.threshold)
                opt_reward = 0.0 if subgoal_done else -1.0

                # store experiences
                self.actor_replay_buffer.store_experience(
                    new_option, state, subgoal, action, next_state, next_state, act_reward, 1 - int(subgoal_done)
                )
                self.optor_replay_buffer.store_experience(
                    new_episode,
                    state,
                    goal,
                    option,
                    next_state,
                    next_state,
                    1 - int(subgoal_done),
                    opt_reward,
                    1 - int(time_done),
                )

                state = next_state
                new_episode = False
                new_option = False

        return next_state, time_done

    def update(self, n_iter, batch_size):
        self.actor.update(self.actor_replay_buffer, n_iter, batch_size)

    def save(self, directory, name):
        self.actor.save(directory, name + "_level_{}".format(0))

    def load(self, directory, name):
        self.actor.load(directory, name + "_level_{}".format(0))

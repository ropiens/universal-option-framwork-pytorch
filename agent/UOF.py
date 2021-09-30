import time
from collections import namedtuple

import numpy as np
import torch

from .DDPG import DDPG
from .DIOL import DIOL
from .utils import (AutoAdjustingConstantChance, ConstantChance, ExpDecayGreedy, HighLevelHindsightReplayBuffer,
                    LowLevelHindsightReplayBuffer)

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
        env,
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
        tau,
        use_aaes=True,
    ):
        self.env = env

        """Inter-Option/High-Level policies - DIOL"""
        # opt, optor refer to high-level policy
        optor_mem_capacity = int(1e5)
        self.optor = DIOL(env, state_dim, option_dim, lr, gamma, tau)
        self.optor_replay_buffer = HighLevelHindsightReplayBuffer(optor_mem_capacity, OPT_Tr)
        self.optor.optor_exploration = ExpDecayGreedy(start=1.0, end=0.02, decay=30000)

        """Intra-Option/Low-Level policies - DDPG + HER (using double critic)"""
        # act, actor refer to low-level policy
        actor_mem_capacity = int(1e5)
        self.actor = DDPG(env, state_dim, action_dim, action_bounds, action_offset, lr, gamma, tau)
        self.actor_replay_buffer = LowLevelHindsightReplayBuffer(actor_mem_capacity, ACT_Tr)

        # Exploration
        self.actor.noise_deviation = 0.05
        if not use_aaes:
            self.actor.actor_exploration = ConstantChance(chance=0.2)
            self.actor.use_aaes = False
        else:
            self.actor.actor_exploration = AutoAdjustingConstantChance(goal_num=option_dim, chance=0.2, tau=0.5)
            self.actor.use_aaes = True

        # set some parameters
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.threshold = threshold
        self.render = render

        self.traning_ep_count = 0

        # logging parameters
        self.goals = [None]
        self.reward = 0
        self.timestep = 0

    def check_goal(self, state, goal, threshold):
        for i in range(self.state_dim):
            if abs(state[i] - goal[i]) > threshold[i]:
                return False
        return True

    def set_subgoal(self, option_ind, option_num):
        subgoals = np.linspace(-1.1, 0.45, option_num)
        return np.array([subgoals[option_ind], 0.04])

    def run_UOF(self, state, goal, option_num=5):
        next_state = None
        time_done = False
        new_episode = True
        # logging updates
        self.goals[0] = goal

        while not time_done:
            subgoal_done = False
            new_option = True
            self.actor.plt_clear()

            # get subgoal from optor
            option = self.optor.select_option(state, goal, ep=self.traning_ep_count)
            subgoal = self.set_subgoal(option, option_num)
            while (not time_done) and (not subgoal_done):
                if self.render:
                    self.env.unwrapped.render_goal(goal, subgoal)

                # take action from actor
                action = self.actor.select_action(state, subgoal, option)
                next_state, rew, time_done, _ = self.env.step(action)
                achieved_goal = next_state

                # check if goal is achieved
                subgoal_done = self.check_goal(achieved_goal, subgoal, self.threshold)
                goal_done = self.check_goal(achieved_goal, goal, self.threshold)
                act_reward = 0.0 if subgoal_done else -1.0
                opt_reward = 0.0 if goal_done else -1.0
                if subgoal_done or goal_done:
                    print(f"subgoal_done: {subgoal_done}, goal_done: {goal_done}")

                # this is for logging
                self.reward += act_reward
                self.timestep += 1

                # store experiences
                self.actor_replay_buffer.store_experience(
                    new_option,
                    state,
                    subgoal,
                    action,
                    next_state,
                    achieved_goal,
                    act_reward,
                    int(subgoal_done),
                )
                self.optor_replay_buffer.store_experience(
                    new_episode,
                    state,
                    goal,
                    option,
                    next_state,
                    achieved_goal,
                    int(subgoal_done),
                    opt_reward,
                    int(time_done),
                )

                state = next_state
                new_episode = False
                new_option = False

        return next_state, time_done

    def update(self, n_iter, batch_size):
        self.actor.update(self.actor_replay_buffer, n_iter, batch_size)
        self.optor.update(self.optor_replay_buffer, n_iter, batch_size)

    def save(self, directory, name):
        self.actor.save(directory, name + "_level_{}".format(0))
        self.optor.save(directory, name + "_level_{}".format(0))

    def load(self, directory, name):
        self.actor.load(directory, name + "_level_{}".format(0))
        self.optor.save(directory, name + "_level_{}".format(0))

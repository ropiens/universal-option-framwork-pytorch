import numpy as np
import torch

from utils import ReplayBuffer

from .DDPG import DDPG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class UOF:
    def __init__(
        self,
        H,
        state_dim,
        action_dim,
        render,
        threshold,
        action_bounds,
        action_offset,
        state_bounds,
        state_offset,
        lr,
    ):

        # adding lowest level
        self.UOF = [DDPG(state_dim, action_dim, action_bounds, action_offset, lr, H)]
        self.replay_buffer = [ReplayBuffer()]

        # set some parameters
        self.H = H
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.threshold = threshold
        self.render = render

        # logging parameters
        self.goals = [None]
        self.reward = 0
        self.timestep = 0

    def set_parameters(
        self,
        lamda,
        gamma,
        action_clip_low,
        action_clip_high,
        state_clip_low,
        state_clip_high,
        exploration_action_noise,
        exploration_state_noise,
    ):

        self.lamda = lamda
        self.gamma = gamma
        self.action_clip_low = action_clip_low
        self.action_clip_high = action_clip_high
        self.state_clip_low = state_clip_low
        self.state_clip_high = state_clip_high
        self.exploration_action_noise = exploration_action_noise
        self.exploration_state_noise = exploration_state_noise

    def check_goal(self, state, goal, threshold):
        for i in range(self.state_dim):
            if abs(state[i] - goal[i]) > threshold[i]:
                return False
        return True

    def run_UOF(self, env, state, goal, is_subgoal_test):
        next_state = None
        done = None
        goal_transitions = []

        # logging updates
        self.goals[0] = goal

        # H attempts
        for _ in range(self.H):
            # if this is a subgoal test, then next/lower level goal has to be a subgoal test
            is_next_subgoal_test = is_subgoal_test

            action = self.UOF[0].select_action(state, goal)

            #   <================ low level policy ================>
            # add noise or take random action if not subgoal testing
            if not is_subgoal_test:
                if np.random.random_sample() > 0.2:
                    action = action + np.random.normal(
                        0, self.exploration_action_noise
                    )
                    action = action.clip(
                        self.action_clip_low, self.action_clip_high
                    )
                else:
                    action = np.random.uniform(
                        self.action_clip_low, self.action_clip_high
                    )

            # take primitive action
            next_state, rew, done, _ = env.step(action)

            if self.render:
                env.render()

                for _ in range(1000000):
                    continue

            # this is for logging
            self.reward += rew
            self.timestep += 1

            #   <================ finish one step/transition ================>

            # check if goal is achieved
            goal_achieved = self.check_goal(next_state, goal, self.threshold)

            # hindsight action transition
            if goal_achieved:
                self.replay_buffer[0].add(
                    (state, action, 0.0, next_state, goal, 0.0, float(done))
                )
            else:
                self.replay_buffer[0].add(
                    (state, action, -1.0, next_state, goal, self.gamma, float(done))
                )

            # copy for goal transition
            goal_transitions.append(
                [state, action, -1.0, next_state, None, self.gamma, float(done)]
            )

            state = next_state

            if done or goal_achieved:
                break

        #   <================ finish H attempts ================>

        # hindsight goal transition
        # last transition reward and discount is 0
        goal_transitions[-1][2] = 0.0
        goal_transitions[-1][5] = 0.0
        for transition in goal_transitions:
            # last state is goal for all transitions
            transition[4] = next_state
            self.replay_buffer[0].add(tuple(transition))

        return next_state, done

    def update(self, n_iter, batch_size):
        self.UOF[0].update(self.replay_buffer[0], n_iter, batch_size)

    def save(self, directory, name):
        self.UOF[0].save(directory, name + "_level_{}".format(0))

    def load(self, directory, name):
        self.UOF[0].load(directory, name + "_level_{}".format(0))

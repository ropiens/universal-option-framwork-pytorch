# origianl code : https://github.com/IanYangChina/UOF-paper-code/blob/main/agent/utils/replay_buffer.py

import random as R
import numpy as np
from copy import deepcopy as dcp


class EpisodeWiseReplayBuffer(object):
    def __init__(self, capacity, tr_namedtuple, seed=0):
        R.seed(seed)
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.episodes = []
        self.ep_position = -1
        self.Transition = tr_namedtuple

    def store_experience(self, new_episode=False, *args):
        # $new_episode is a boolean value
        if new_episode:
            self.episodes.append([])
            self.ep_position += 1
        self.episodes[self.ep_position].append(self.Transition(*args))

    def store_episode(self):
        if len(self.episodes) == 0:
            return
        for ep in self.episodes:
            for n in range(len(ep)):
                if len(self.memory) < self.capacity:
                    self.memory.append(None)
                self.memory[self.position] = ep[n]
                self.position = (self.position + 1) % self.capacity
        self.episodes.clear()
        self.ep_position = -1

    def sample(self, batch_size):
        batch = R.sample(self.memory, batch_size)
        return self.Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)


class LowLevelHindsightReplayBuffer(EpisodeWiseReplayBuffer):
    def __init__(self, capacity, tr_namedtuple, sampled_goal_num=4, seed=0):
        self.k = sampled_goal_num
        EpisodeWiseReplayBuffer.__init__(self, capacity, tr_namedtuple, seed)

    def modify_experiences(self):
        if len(self.episodes) == 0:
            return
        for _ in range(len(self.episodes)):
            ep = self.episodes[_]
            goals = self.sample_achieved_goal_random(ep)
            for n in range(len(goals[0])):
                ind = goals[0][n]
                goal = goals[1][n]
                modified_ep = []

                for tr in range(ind + 1):
                    s = ep[tr].state
                    dg = goal
                    a = ep[tr].action
                    ns = ep[tr].next_state
                    ag = ep[tr].achieved_goal
                    r = ep[tr].reward
                    d = ep[tr].done
                    if tr == ind:
                        modified_ep.append(self.Transition(s, dg, a, ns, ag, 0.0, 0))
                    else:
                        modified_ep.append(self.Transition(s, dg, a, ns, ag, r, d))
                self.episodes.append(modified_ep)

    def sample_achieved_goal_random(self, ep):
        goals = [[], []]
        for k_ in range(self.k):
            done = False
            count = 0
            while not done:
                count += 1
                if count > len(ep):
                    break
                ind = R.randint(0, len(ep) - 1)
                goal = ep[ind].achieved_goal
                if all(not np.array_equal(goal, g) for g in goals[1]):
                    goals[1].append(goal)
                    goals[0].append(ind)
                    done = True
        return goals


class HighLevelHindsightReplayBuffer(EpisodeWiseReplayBuffer):
    def __init__(self, capacity, tr_namedtuple, modified_ep_num=3, seed=0):
        EpisodeWiseReplayBuffer.__init__(self, capacity, tr_namedtuple, seed)
        self.modified_ep_num = modified_ep_num

    def modify_experiences(self):
        if len(self.episodes) == 0:
            return
        for _ in range(len(self.episodes)):
            ep = self.episodes[_]
            # if any option has been achieved
            # the whole trajectory before that option is done
            # could have been rewarded if the achieved goal was the desired goal
            count = 0
            for ind in range(len(ep)):
                if not (1 - ep[ind].option_done):
                    continue
                else:
                    if ep[ind].reward == 0.0:
                        continue
                    else:
                        modified_ep = []
                        for tr in range(ind + 1):
                            s = ep[tr].state
                            dg = ep[ind].achieved_goal
                            o = ep[tr].option
                            ns = ep[tr].next_state
                            op_d = ep[tr].option_done
                            ag = ep[tr].achieved_goal
                            ts = ep[tr].timesteps
                            r = ep[tr].reward
                            d = ep[tr].done
                            if tr == ind:
                                modified_ep.append(self.Transition(s, dg, o, ns, ag, op_d, ts, 0.0, d))
                            else:
                                modified_ep.append(self.Transition(s, dg, o, ns, ag, op_d, ts, r, d))
                        self.episodes.append(dcp(modified_ep))
                        count += 1
                        if count >= self.modified_ep_num:
                            break

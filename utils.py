import numpy as np


class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0

    def add(self, transition):
        assert len(transition) == 7, "transition must have length = 7"

        # transiton is tuple of (state, action, reward, next_state, goal, achieved_goal, done)
        self.buffer.append(transition)
        self.size += 1

    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0 : int(self.size / 5)]
            self.size = len(self.buffer)

        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, goals, achieved_goals, dones = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for i in indexes:
            states.append(np.array(self.buffer[i][0], copy=False))
            actions.append(np.array(self.buffer[i][1], copy=False))
            rewards.append(np.array(self.buffer[i][2], copy=False))
            next_states.append(np.array(self.buffer[i][3], copy=False))
            goals.append(np.array(self.buffer[i][4], copy=False))
            achieved_goals.append(np.array(self.buffer[i][5], copy=False))
            dones.append(np.array(self.buffer[i][6], copy=False))

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(goals),
            np.array(achieved_goals),
            np.array(dones),
        )

class LowLevelHindsightReplayBuffer(ReplayBuffer):
    def __init__(self, max_size=5e5, sampled_goal_num=4):
        self.k = sampled_goal_num
        ReplayBuffer.__init__(self, max_size)

    def modify(self):
        if len(self.buffer) == 0:
            return

        for i in range(len(self.buffer)):
            ep = self.buffer[i]
            goals = self.sample_achieved_goal_random(ep)
            for n in range(len(goals[0])):
                ind = goals[0][n]
                goal = goals[1][n]
                modified_ep = []

                for tr in range(ind+1):
                    state = ep[tr][0]
                    action = ep[tr][1]
                    reward = ep[tr][2]
                    next_state = ep[tr][3]
                    goal = ep[tr][4]
                    achieved_goal = ep[tr][5]
                    done = ep[tr][6]
                    if tr == ind:
                        modified_ep.append((state, action, 0.0, next_state, goal, achieved_goal, 0))
                    else:
                        modified_ep.append((state, action, reward, next_state, goal, achieved_goal, done))

    def sample_achieved_goal_random(self, ep):
        goals = [[],[]]
        for k_ in range(self.k):
            done = False
            count = 0
            while not done:
                count +=1
                if count > len(ep):
                    break
                ind = np.random.randint(0, len(ep)-1)
                print(len(ep), ind)
                print(ep[int(ind)])
                goal = ep[int(ind)][5]
                if all(not np.array_equal(goal, g) for g in goals[1]):
                    goals[1].append(goal)
                    goals[0].append(ind)
                    done =True
        return goals


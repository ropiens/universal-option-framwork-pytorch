import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_bounds, offset):
        super(Actor, self).__init__()
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim + state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh(),
        )
        # max value of actions
        self.action_bounds = action_bounds
        self.offset = offset

    def forward(self, state, goal):
        return (self.actor(torch.cat([state, goal], 1)) * self.action_bounds) + self.offset


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # UVFA critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim + state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, state, action, goal):
        return -self.critic(torch.cat([state, action, goal], 1))


class DDPG:
    def __init__(self, env, state_dim, action_dim, action_bounds, offset, lr, gamma, tau):
        self.env = env
        self.gamma = gamma
        self.tau = tau

        self.action_dim = action_dim
        self.action_max = action_bounds + offset
        self.action_min = -action_bounds + offset

        self.actor = Actor(state_dim, action_dim, action_bounds, offset).to(device)
        self.target_actor = Actor(state_dim, action_dim, action_bounds, offset).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic_1 = Critic(state_dim, action_dim).to(device)
        self.target_critic_1 = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer_1 = optim.Adam(self.critic_1.parameters(), lr=lr)

        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.target_critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer_2 = optim.Adam(self.critic_2.parameters(), lr=lr)

        self.soft_update(tau=1.0)

        self.mseLoss = torch.nn.MSELoss()

        # for exploration
        self.use_aaes = False
        self.actor_exploration = None
        self.noise_deviation = None

    def select_action(self, state, goal, step):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        goal = torch.FloatTensor(goal.reshape(1, -1)).to(device)

        action = self.target_actor(state, goal).detach().cpu().data.numpy().flatten()

        # To Do: add test mode
        if self.env.np_random.uniform(0, 1) < self.actor_exploration(step):
            action = self.env.np_random.uniform(self.action_min, self.action_max, size=(self.action_dim,))
        else:
            if self.use_aaes:
                deviation = self.noise_deviation * (1 - self.actor_exploration.success_rates[step])
            else:
                deviation = self.noise_deviation
            action += deviation * self.env.np_random.randn(self.action_dim)
            action = np.clip(action, self.action_min, self.action_max).detach().cpu().data.numpy().flatten()
        return action

    def soft_update(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def update(self, buffer, n_iter, batch_size):
        if len(buffer.episodes) == 0:
            return

        # modify experiences in hindsight
        buffer.modify_experiences()
        buffer.store_episode()

        if len(buffer) < batch_size:
            return

        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            batch = buffer.sample(batch_size)
            state, action, reward, next_state, goal, achieved_goal, done = (
                batch.state,
                batch.action,
                batch.reward,
                batch.next_state,
                batch.desired_goal,
                batch.achieved_goal,
                batch.done,
            )

            # convert np arrays into tensors
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size, 1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            goal = torch.FloatTensor(goal).to(device)
            achieved_goal = torch.FloatTensor(achieved_goal).to(device)
            done = torch.FloatTensor(done).reshape((batch_size, 1)).to(device)

            # select next action
            next_action = self.target_actor(next_state, goal).detach()

            # Compute target Q-value:
            value_1 = self.target_critic_1(next_state, next_action, goal).detach()
            value_2 = self.target_critic_2(next_state, next_action, goal).detach()
            target_Q = torch.min(value_1, value_2)
            target_Q = reward + ((1 - done) * self.gamma * target_Q)

            # Optimize Critic:
            critic_loss_1 = self.mseLoss(self.critic_1(state, action, goal), target_Q)
            self.critic_optimizer_1.zero_grad()
            critic_loss_1.backward()
            self.critic_optimizer_1.step()

            critic_loss_2 = self.mseLoss(self.critic_2(state, action, goal), target_Q)
            self.critic_optimizer_2.zero_grad()
            critic_loss_2.backward()
            self.critic_optimizer_2.step()

            # Compute actor loss:
            new_value_1 = self.critic_1(state, self.actor(state, goal), goal)
            new_value_2 = self.critic_1(state, self.actor(state, goal), goal)
            actor_loss = -torch.min(new_value_1, new_value_2).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update()

    def save(self, directory, name):
        torch.save(self.actor.state_dict(), "%s/%s_actor.pth" % (directory, name))
        torch.save(self.critic_1.state_dict(), "%s/%s_crtic_1.pth" % (directory, name))
        torch.save(self.critic_2.state_dict(), "%s/%s_crtic_2.pth" % (directory, name))

    def load(self, directory, name):
        self.actor.load_state_dict(torch.load("%s/%s_actor.pth" % (directory, name), map_location="cpu"))
        self.critic_1.load_state_dict(torch.load("%s/%s_crtic_1.pth" % (directory, name), map_location="cpu"))
        self.critic_2.load_state_dict(torch.load("%s/%s_crtic_2.pth" % (directory, name), map_location="cpu"))

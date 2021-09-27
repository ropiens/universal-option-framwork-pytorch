import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Mlp(nn.Module):
    def __init__(self, state_dim, option_dim, fc1_size=64, fc2_size=64, init_w=3e-3):
        super(Mlp, self).__init__()
        self.input_dim = state_dim + state_dim  # state + goal
        self.output_dim = option_dim
        self.fc1 = nn.Linear(self.input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.v = nn.Linear(fc2_size, self.output_dim)
        # initialize weight and bias of the final layer to make near-0 outputs
        self.v.weight.data.uniform_(-init_w, init_w)
        self.v.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, goal):
        x = F.relu(self.fc1(torch.cat([state, goal], 1)))
        x = F.relu(self.fc2(x))
        q_values = self.v(x)
        return q_values


class DIOL:
    def __init__(self, env, state_dim, option_dim, lr, gamma, tau):
        self.env = env
        self.gamma = gamma
        self.tau = tau
        self.clip_value = -100
        self.option_num = option_dim

        self.optor = Mlp(state_dim, option_dim).to(device)
        self.target_optor = Mlp(state_dim, option_dim).to(device)
        self.optor_optimizer = optim.Adam(self.optor.parameters(), lr=lr)

        self.optor_2 = Mlp(state_dim, option_dim).to(device)
        self.target_optor_2 = Mlp(state_dim, option_dim).to(device)
        self.optor_optimizer_2 = optim.Adam(self.optor_2.parameters(), lr=lr)

        self.soft_update(tau=1.0)

    def soft_update(self, tau=None):
        if tau is None:
            tau = self.tau

        for target_param, param in zip(self.target_optor.parameters(), self.optor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        for target_param, param in zip(self.target_optor_2.parameters(), self.optor_2.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def select_option(self, state, high_level_goal, ep=0):

        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        high_level_goal = torch.FloatTensor(high_level_goal.reshape(1, -1)).to(device)

        option_values = self.target_optor(state, high_level_goal)

        if self.env.np_random.uniform(0, 1) < self.optor_exploration(ep):
            option = self.env.np_random.randint(0, self.option_num - 1)
        else:
            option = torch.argmax(option_values).item()
        return option

    def update(self, buffer, n_iter, batch_size):
        if len(buffer.episodes) == 0:
            return

        buffer.store_episode()

        if len(buffer) < batch_size:
            return

        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            batch = buffer.sample(batch_size)
            state, option, reward, next_state, goal, achieved_goal, option_done, done = (
                batch.state,
                batch.option,
                batch.reward,
                batch.next_state,
                batch.desired_goal,
                batch.achieved_goal,
                batch.option_done,
                batch.done,
            )

            # convert np arrays into tensors
            state = torch.FloatTensor(state).to(device)
            option = torch.LongTensor(option).unsqueeze(1).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size, 1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            goal = torch.FloatTensor(goal).to(device)
            achieved_goal = torch.FloatTensor(achieved_goal).to(device)
            option_done = torch.FloatTensor(option_done).reshape((batch_size, 1)).to(device)
            done = torch.FloatTensor(done).reshape((batch_size, 1)).to(device)

            # calculate "option value upon arrival"
            with torch.no_grad():
                unchanged_next_option_values = self.target_optor(next_state, goal).gather(1, option)
                maximal_next_option_values = self.target_optor(next_state, goal).max(1)[0].view(batch_size, 1)
                next_option_values = (
                    option_done * maximal_next_option_values + (1 - option_done) * unchanged_next_option_values
                )

                unchanged_next_option_values_2 = self.target_optor_2(next_state, goal).gather(1, option)
                maximal_next_option_values_2 = self.target_optor_2(next_state, goal).max(1)[0].view(batch_size, 1)
                next_option_values_2 = (
                    option_done * maximal_next_option_values_2
                    + (1 - option_done) * unchanged_next_option_values_2
                )

            next_option_values = torch.min(next_option_values, next_option_values_2)
            target_option_values = reward + (1 - done) * self.gamma * next_option_values
            target_option_values = torch.clamp(target_option_values, self.clip_value, -0.0)

            self.optor_optimizer.zero_grad()
            estimated_option_values = self.target_optor(state, goal).gather(1, option)
            loss = F.smooth_l1_loss(estimated_option_values, target_option_values.detach())
            loss.backward()
            self.optor_optimizer.step()

            self.optor_optimizer_2.zero_grad()
            estimated_option_values_2 = self.target_optor_2(state, goal).gather(1, option)
            loss_2 = F.smooth_l1_loss(estimated_option_values_2, target_option_values.detach())
            loss_2.backward()
            self.optor_optimizer_2.step()

            self.soft_update(tau=1.0)

    def save(self, directory, name):
        torch.save(self.target_optor.state_dict(), "%s/%s_optor.pth" % (directory, name))
        torch.save(self.optor_2.state_dict(), "%s/%s_optor_2.pth" % (directory, name))

    def load(self, directory, name):
        self.target_optor.load_state_dict(torch.load("%s/%s_optor.pth" % (directory, name), map_location="cpu"))
        self.optor_2.load_state_dict(torch.load("%s/%s_optor_2.pth" % (directory, name), map_location="cpu"))

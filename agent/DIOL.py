import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Mlp(nn.Module):
    def __init__(self, input_dims, output_dims, fc1_size=64, fc2_size=64, init_w=3e-3):
        super(Mlp, self).__init__()
        self.input_dim = input_dims
        self.output_dim = output_dims
        self.fc1 = nn.Linear(self.input_dim, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.v = nn.Linear(fc2_size, self.output_dim)
        # initialize weight and bias of the final layer to make near-0 outputs
        self.v.weight.data.uniform_(-init_w, init_w)
        self.v.bias.data.uniform_(-init_w, init_w)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q_values = self.v(x)
        return q_values


class DIOL:
    def __init__(self, state_dim, option_dim, lr, gamma):

        self.gamma = gamma

        self.optor_1 = Mlp(state_dim, option_dim).to(device)
        self.optor_optimizer_1 = optim.Adam(self.actor.parameters(), lr=lr)

        self.optor_2 = Mlp(state_dim, option_dim).to(device)
        self.optor_optimizer_2 = optim.Adam(self.actor.parameters(), lr=lr)

        self.mseLoss = torch.nn.MSELoss()

    def select_option(self, state, high_level_goal, desired_goal_id, ep=0):
        pass

    def update(self, buffer):
        pass

    def save(self, directory, name):
        torch.save(self.actor.state_dict(), "%s/%s_optor_1.pth" % (directory, name))
        torch.save(self.critic_1.state_dict(), "%s/%s_optor_2.pth" % (directory, name))

    def load(self, directory, name):
        self.optor_1.load_state_dict(torch.load("%s/%s_optor_1.pth" % (directory, name), map_location="cpu"))
        self.optor_2.load_state_dict(torch.load("%s/%s_optor_2.pth" % (directory, name), map_location="cpu"))

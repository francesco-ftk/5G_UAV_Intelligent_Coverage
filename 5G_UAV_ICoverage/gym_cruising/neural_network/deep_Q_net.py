import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_sizes = [256, 256]


class DeepQNet(nn.Module):
    def __init__(self, state_dim=16, action_dim=2):
        super(DeepQNet, self).__init__()
        self.fl1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
        self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fl3 = nn.Linear(hidden_sizes[1], 1)  # Output Q(s, a)

    def forward(self, state, action):
        combined = torch.cat([state, action], dim=1)
        # Process state through its branch
        out = F.relu(self.fl1(combined))
        out = F.relu(self.fl2(out))
        # return Q-value(s, a)
        return self.fl3(out)

    # TODO
# class DoubleDeepQNet(nn.Module):
#     def __init__(self, state_dim=16, action_dim=2):
#         super(DoubleDeepQNet, self).__init__()
#         self.fl1 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
#         self.fl2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
#         self.fl3 = nn.Linear(hidden_sizes[1], 1)  # Output Q1(s, a)
#
#         self.fl4 = nn.Linear(state_dim + action_dim, hidden_sizes[0])
#         self.fl5 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
#         self.fl6 = nn.Linear(hidden_sizes[1], 1)  # Output Q2(s, a)
#
#     def forward(self, state, action):
#         combined = torch.cat([state, action], dim=1)
#         # Process state through its branch
#         out = F.relu(self.fl1(combined))
#         out = F.relu(self.fl2(out))
#         # return Q-value(s, a)
#
#         out1 = F.relu(self.fl4(combined))
#         out1 = F.relu(self.fl5(out1))
#
#         return self.fl3(out), self.fl6(out1)

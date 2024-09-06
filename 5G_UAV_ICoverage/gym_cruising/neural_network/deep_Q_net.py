import torch
import torch.nn as nn
import torch.nn.functional as F

hidden_sizes = [8, 4, 2]

class DeepQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DeepQNet, self).__init__()

        # Branch for state input
        self.fl1_state = nn.Linear(state_dim, hidden_sizes[0])
        self.fl2_state = nn.Linear(hidden_sizes[0], hidden_sizes[1])

        # Branch for action input (this will be concatenated later)
        self.fl1_action = nn.Linear(action_dim, hidden_sizes[2])

        # Combined layers after concatenating state and action
        self.fl3 = nn.Linear(hidden_sizes[1] + hidden_sizes[2], hidden_sizes[1])
        self.fl4 = nn.Linear(hidden_sizes[1], 1)  # Output Q(s, a)

    def forward(self, state, action):
        # Process state through its branch
        state_out = F.relu(self.fl1_state(state))
        state_out = F.relu(self.fl2_state(state_out))

        # Process action through its branch
        action_out = F.relu(self.fl1_action(action))

        # Concatenate state and action outputs
        combined = torch.cat([state_out, action_out], dim=1)

        # Further processing
        x = F.relu(self.fl3(combined))

        # return Q-value(s, a)
        return self.fl4(x)

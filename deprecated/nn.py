import torch
import torch.nn as nn
import torch.nn.functional as F

#
# class CNNForEnv(nn.Module):
#     def __init__(self, state_dim: int, action_dim: int, activation=F.relu):
#         super(CNNForEnv, self).__init__()
#         self.conv1 = nn.Conv2d(state_dim, 16, kernel_size=8, stride=4)  # [N, 4, 84, 84] -> [N, 16, 20, 20]
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)  # [N, 16, 20, 20] -> [N, 32, 9, 9]
#         # self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
#         self.in_features = 32 * 9 * 9
#         self.fc1 = nn.Linear(self.in_features, 256)
#         # self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(256, action_dim)
#         self.activation = activation
#
#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         x = F.relu(self.conv2(x))
#         # x = F.relu(self.conv3(x))
#         x = x.view((-1, self.in_features))
#         x = self.fc1(x)
#         # x = self.fc2(x)
#         x = self.fc3(x)
#         return x


class CNNForEnv(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, activation=F.relu):
        super().__init__()

        self.conv1 = nn.Conv2d(state_dim, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.linear1 = nn.Linear(32 * 9 * 9, 32)
        self.policy = nn.Linear(32, action_dim)
        # self.value = nn.Linear(256, 1)

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))

        flattened = torch.flatten(conv2_out, start_dim=1)  # N x 9*9*32
        linear1_out = self.linear1(flattened)

        policy_output = self.policy(linear1_out)
        # value_output = self.value(linear1_out)

        # probs = F.softmax(policy_output)
        # log_probs = F.log_softmax(policy_output)
        return policy_output

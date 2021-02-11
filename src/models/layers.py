# External imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# A Basis for so(3):
L_0 = torch.tensor([[0, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]], dtype=torch.float)

L_1 = torch.tensor([[0, 0, 1],
                    [0, 0, 0],
                    [-1, 0, 0]], dtype=torch.float)

L_2 = torch.tensor([[0, -1, 0],
                    [1, 0, 0],
                    [0, 0, 0]], dtype=torch.float)

class SO3Block(nn.Module):
    def __init__(self, euclidean_network):
        super(SO3Block, self).__init__()
        self.register_buffer("id", torch.eye(3, dtype=torch.float))
        self.register_buffer("generators", torch.stack([L_0, L_1, L_2]))

        self.euclidean_network = euclidean_network

    def linear_combo(self, alg_coefs):
        # Return linear combo of so(3) matrix generator

        # Remove batch dimension
        alg_coefs = alg_coefs.squeeze(0)

        # Compute linear combination of algebre generators specified by alg_coefs
        terms = [torch.mul(alg_coefs[i], self.generators[i]) for i in range(len(alg_coefs))]
        linear_combo = torch.stack(terms).sum(dim=0)

        return linear_combo

    def exponential_map(self, linear_combo):
        # Using Rodriguez Formula
        two_norm = torch.linalg.norm(linear_combo)
        normalized = linear_combo / two_norm
        g = self.id + normalized*torch.sin(two_norm) + torch.matrix_power(normalized, 2)*(1 - torch.cos(two_norm))
        return g

    def forward(self, x, c):
        # Note: batch size of 1 required at the moment!
        alg_coefs = self.euclidean_network(x)
        linear_combo = self.linear_combo(alg_coefs)
        g = self.exponential_map(linear_combo)
        output = torch.matmul(g, c)
        return output


class FF(nn.Module):
    def __init__(self):
        super(FF, self).__init__()
        self.l1 = nn.Linear(10, 3)
        self.l2 = nn.Linear(3, 3)

    def forward(self, x):
        out = self.l1(x)
        out = F.relu(out)
        out = self.l2(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, in_channels, in_size):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 5, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128*((((in_size - 4)//2) - 4)**2), 128)
        self.fc2 = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output


class ClassificationHead(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationHead, self).__init__()
        self.l1 = nn.Linear(3, num_classes)

    def forward(self, x):
        output = self.l1(x)
        return output

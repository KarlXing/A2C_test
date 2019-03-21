import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNAutoEncoder(nn.Module):
    def __init__(self, num_inputs, hidden_size):
        super(CNNAutoEncoder, self).__init__()
        init_relu = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        init_sigmoid = lambda m: init(m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('sigmoid'))

        # convolution
        self.conv1 = init_relu(nn.Conv2d(num_inputs, 32, 8, stride=4))
        self.conv2 = init_relu(nn.Conv2d(32, 64, 4, stride=2))
        self.conv3 = init_relu(nn.Conv2d(64, 32, 3, stride=1))
        self.f1 = init_sigmoid(nn.Linear(1568, hidden_size))
        # deconvolution
        self.df1 = init_relu(nn.Linear(hidden_size, 1568))
        self.dc1 = init_relu(nn.ConvTranspose2d(32, 64, 3, stride = 1))
        self.dc2 = init_relu(nn.ConvTranspose2d(64, 32, 4, stride = 2))
        self.dc3 = init_sigmoid(nn.ConvTranspose2d(32, 1, 8, stride = 4))


    def encode(self, inputs, noise = None):
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        if noise is None:
            x = F.sigmoid(self.f1(x))
        else:
            x = F.sigmoid(self.f1(x) + noise)
        return x

    def decode(self, repre):
        x = F.relu(self.df1(repre))
        x = x.view(x.size(0), 32, 7, 7)
        x = F.relu(self.dc1(x))
        x = F.relu(self.dc2(x))
        x = F.sigmoid(self.dc3(x))
        return x

    def forward(self, inputs, noise):
        repre = self.encode(inputs, noise)
        x = self.decode(repre)
        return x, repre


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from rl_trainer.algo.popart import PopArt


class CNN_encoder(nn.Module):
    def __init__(self):
        super(CNN_encoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 2),
            nn.Conv2d(8, 8, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(4, 2),
            nn.Flatten(),
        )

    def forward(self, view_state):
        x = self.net(view_state)
        return x


class Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64, cnn=False):
        super(Actor, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_encoder()
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob


class Critic(nn.Module):
    def __init__(self, state_space, hidden_size=64, cnn=False,task = 1,popart=False,device="cpu"):
        super(Critic, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_encoder()
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.state_value = nn.Linear(hidden_size, 1) if not popart else PopArt(hidden_size,task,device=device)

    def forward(self, x):
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        value = self.state_value(x)
        return value


class CNN_Actor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64,use_onehot=False):
        super(CNN_Actor, self).__init__()
        
        self.use_onehot = use_onehot
        in_channels = 8 if use_onehot else 1

        # self.conv1 = nn.Conv2d(in_channels = 8, out_channels=32, kernel_size = 4, stride = 2)
        # self.conv2 = nn.Conv2d(in_channels = 32, out_channels=64, kernel_size = 3, stride = 1)
        # self.flatten = nn.Flatten()
        self.net = Net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
        )

        self.linear1 = nn.Linear(256, 64)
        self.linear2 = nn.Linear(64, action_space)

    def forward(self, x):
        if self.use_onehot:
            x = F.one_hot(x, 8)
            x = torch.reshape(x,(-1,8,25,25))
        else:
            x = torch.reshape(x,(-1,1,25,25))
        x = self.net(x)
        x = torch.relu(self.linear1(x))
        action_prob = F.softmax(self.linear2(x), dim=-1)
        return action_prob


class CNN_Critic(nn.Module):
    def __init__(self, state_space, hidden_size=64,use_onehot=False):
        super(CNN_Critic, self).__init__()
        
        self.use_onehot = use_onehot
        in_channels = 8 if use_onehot else 1

        self.net = Net = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
        )

        self.linear1 = nn.Linear(256, 64)
        self.linear2 = nn.Linear(64, 1)

    def forward(self, x):
        if self.use_onehot:
            x = F.one_hot(x, 8)
            x = torch.reshape(x,(-1,8,25,25))
        else:
            x = torch.reshape(x,(-1,1,25,25))
        x = self.net(x)
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class RNN_Actor(nn.Module):
    def __init__(self,state_space,action_space,hidden_size=64):
        super(RNN_Actor,self).__init__()
        
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.rnn = nn.GRU(hidden_size,hidden_size,num_layers=1)
        self.action_head = nn.Linear(hidden_size, action_space)
        
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        self.rnn_norm = nn.LayerNorm(hidden_size)
        
    def forward(self,x,rnn_state):
        self.rnn.flatten_parameters()
        T,B = x.shape[:2]
        x = F.relu(self.linear_in(x.flatten(0,1))).view(T,B,-1)
        x,rnn_state = self.rnn(x,rnn_state)
        x = self.rnn_norm(x)
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob,rnn_state
    
class RNN_Critic(nn.Module):
    def __init__(self,state_space,hidden_size=64):
        super(RNN_Critic,self).__init__()
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.rnn = nn.GRU(hidden_size,hidden_size,num_layers=1)
        self.state_value = nn.Linear(hidden_size, 1)
        
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        self.rnn_norm = nn.LayerNorm(hidden_size)
    
    def forward(self,x,rnn_state):
        self.rnn.flatten_parameters()
        x = F.relu(self.linear_in(x))
        x,rnn_state = self.rnn(x,rnn_state)
        x = self.rnn_norm(x)
        value = self.state_value(x)
        return value,rnn_state

class CNN_CategoricalActor(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64):
        super(CNN_CategoricalActor, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
        )

        self.linear1 = nn.Linear(128, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = self.net(x)
        x = F.relu(self.linear1(x))
        action_prob = F.softmax(self.linear2(x), dim=-1)
        c = Categorical(action_prob)
        sampled_action = c.sample()
        greedy_action = torch.argmax(action_prob)
        return sampled_action, action_prob, greedy_action


class CNN_Critic2(nn.Module):
    def __init__(self, state_space, action_space, hidden_size=64):
        super(CNN_Critic2, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=32, kernel_size=4, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Flatten(),
        )
        self.linear1 = nn.Linear(128, hidden_size)
        self.linear2 = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = self.net(x)
        x = F.relu(self.linear1(x))
        return self.linear2(x)

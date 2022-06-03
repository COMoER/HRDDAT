"""
Modified from
https://github.com/semin-park/Large-Scale-Study-of-Curiosity-Driven-Learning
"""
import torch


# To get random features, create a FeatureEncoder instance and use it `with torch.no_grad()`
class FeatureEncoder(torch.nn.Module):
    def __init__(self, c_in):
        super(FeatureEncoder, self).__init__()
        # input size is supposed to be (batch_size, C, 25,25)
        # 25->13->7->4->2
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(c_in, 32, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),

            torch.nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.conv(x)
        x = x.view(B, -1)
        return x


class FeatureEncoderMLP(torch.nn.Module):
    def __init__(self, c_in,c_out):
        super(FeatureEncoderMLP, self).__init__()
        # input size is supposed to be (batch_size, C, 25,25)
        # 25->13->7->4->2
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(c_in,64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,c_out),
            torch.nn.ReLU())


    def forward(self, x):
        # B = x.shape[0]
        x = self.mlp(x)
        # x = x.view(B, -1)
        return x


class InverseModel(torch.nn.Module):
    def __init__(self, num_features, num_actions):
        super(InverseModel, self).__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_actions),
        )

    def forward(self, x_cur, x_next):
        # each input is supposed to be (batch_size, feature_size)
        x = torch.cat([x_cur, x_next], dim=1)
        x = self.fc(x)
        x = torch.softmax(x, dim=1)
        return x


class ForwardModel(torch.nn.Module):
    def __init__(self, num_features, num_actions):
        super(ForwardModel, self).__init__()
        
        self.bn = torch.nn.BatchNorm1d(num_features + num_actions)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(num_features + num_actions, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_features)
        )

    def forward(self, x, action):
        """
        Args:
            x: (T, Rollout_num, num_features)
            action: (T, Rollout_num, num_actions)
        """
        x = torch.cat([x, action], dim=-1)
        x = self.bn(x.permute((1,2,0))).permute((2,0,1)) if x.dim() == 3 else self.bn(x)
        x = self.fc(x)
        return x

class IntrinsicCuriosityModule(torch.nn.Module):
    def __init__(self, c_in, num_actions, use_random_features):
        super(IntrinsicCuriosityModule, self).__init__()
        self.random = use_random_features
        self.encoder = FeatureEncoderMLP(c_in,num_actions)
        if self.random:
            self.encoder.eval()
        else:
            self.inv_model = InverseModel(num_actions, num_actions)
        self.fwd_model = ForwardModel(num_actions , num_actions)

    def forward(self, x0, a, x1):
        # x0, x1: (T, R, 25, 25), a: (T, num_actions)
        if x0.dim() == 4:
            x0 = x0.flatten(2)
        if x1.dim() == 4:
            x1 = x1.flatten(2)

        with torch.set_grad_enabled(not self.random):
            s0 = self.encoder(x0)
            s1 = self.encoder(x1)
        action_pred = self.inv_model(s0, s1) if not self.random else None
        s1_pred = self.fwd_model(s0, a)
        return s1, s1_pred, action_pred


import os
import sys
from os import path
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

father_path = path.dirname(__file__)
sys.path.append(str(os.path.dirname(father_path)))

from rl_trainer.algo.network import Actor, CNN_Actor, CNN_Critic, Critic
from rl_trainer.algo.utils import AvgMeter, RunningMeanStd
from torch.utils.tensorboard import SummaryWriter


class Args:
    gae_lambda = 0.95
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 5
    buffer_capacity = 1000
    batch_size = 32
    gamma = 0.99
    lr = 0.0001

    action_space = 36
    state_space = 625


args = Args()


class PPO:
    clip_param = args.clip_param
    max_grad_norm = args.max_grad_norm
    ppo_update_time = args.ppo_update_time
    buffer_capacity = args.buffer_capacity
    batch_size = args.batch_size
    gamma = args.gamma
    action_space = args.action_space
    state_space = args.state_space
    lr = args.lr
    gae_lambda = args.gae_lambda
    use_cnn = False

    def __init__(
        self,
        device: str = "cpu",
        run_dir: str = None,
        writer: SummaryWriter = None,
        use_gae: bool = True,
    ):
        super(PPO, self).__init__()
        self.args = args
        if self.use_cnn:
            self.actor_net = CNN_Actor(self.state_space, self.action_space)
            self.critic_net = CNN_Critic(self.state_space)
        else:
            self.actor_net = Actor(self.state_space, self.action_space)
            self.critic_net = Critic(self.state_space)
        self.actor_net = self.actor_net.to(device)
        self.critic_net = self.critic_net.to(device)

        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=self.lr)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=self.lr)
        
        self.reward_norm = True #parsed_args.reward_norm
        self.adv_norm = True #parsed_args.advt_norm

        self.device = device

        self.run_dir = run_dir
        self.writer = writer
        self.IO = True if (run_dir is not None) else False

        self.use_gae = use_gae
        
        self.reward_tracker = RunningMeanStd()

    def select_action(self, state, train=True):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        if train:
            action = c.sample()
        else:
            action = torch.argmax(action_prob)
        return action.cpu().item(), action_prob[:, action.item()].item()

    def choose_action(self, state, train=False):
        return self.select_action(state, train)[0]

    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.cpu().item()

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, ep_i):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(
            self.device
        )
        action = (
            torch.tensor([t.action for t in self.buffer], dtype=torch.long)
            .view(-1, 1)
            .to(self.device)
        )
        reward = [t.reward for t in self.buffer]

        old_action_log_prob = (
            torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float)
            .view(-1, 1)
            .to(self.device)
        )
        
        if self.reward_norm:
            reward = torch.Tensor(reward)
            reward_mean, reward_std, count = reward.mean().cpu().detach().numpy(), reward.std().cpu().detach().numpy(), len(
                reward)
            self.reward_tracker.update_from_moments(reward_mean, reward_std ** 2, count)
            std = np.sqrt(self.reward_tracker.var)
            reward /= std

        if self.use_gae:
            value = self.critic_net(state).cpu().detach().reshape(-1)
            R = reward[-1] - value[-1]
            Gt = [R]
            for step_i in range(len(reward) - 2, -1, -1):
                delta = reward[step_i] + self.gamma * value[step_i + 1] - value[step_i]
                R = delta + self.gamma * self.gae_lambda * R
                Gt.insert(0, R)
            Gt = torch.tensor(Gt, dtype=torch.float).to(self.device)
            Advt = Gt.clone()
            Gt += value.to(self.device)
            
            if self.adv_norm:
                Advt = (Advt - torch.mean(Advt, 0, keepdim=True)) / (torch.std(Advt, 0, keepdim=True) + 1e-10)
        else:
            R = 0
            Gt = []
            for r in reward[::-1]:
                R = r + self.gamma * R
                Gt.insert(0, R)
            Gt = torch.tensor(Gt, dtype=torch.float).to(self.device)
        for i in range(self.ppo_update_time):
            for index in BatchSampler(
                SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False
            ):
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index].squeeze(1))
                delta = Gt_index - V
                if self.use_gae:
                    advantage = Advt[index]
                else:
                    advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index].squeeze(1)).gather(
                    1, action[index]
                )  # new policy

                ratio = action_prob / old_action_log_prob[index]
                surr1 = ratio * advantage # [B,B]
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                    * advantage
                )

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.actor_net.parameters(), self.max_grad_norm
                )
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(
                    self.critic_net.parameters(), self.max_grad_norm
                )
                self.critic_net_optimizer.step()
                self.training_step += 1

                if self.IO:
                    self.writer.add_scalar("loss/policy loss", action_loss.item(), ep_i)
                    self.writer.add_scalar("loss/critic loss", value_loss.item(), ep_i)

        self.clear_buffer()

    def clear_buffer(self):
        del self.buffer[:]

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, "trained_model")
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        torch.save(self.actor_net.state_dict(), model_actor_path)
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        torch.save(self.critic_net.state_dict(), model_critic_path)

    def load(self, run_dir, episode):
        print(f"\nBegin to load model: ")
        print("run_dir: ", run_dir)
        base_path = os.path.dirname(os.path.dirname(__file__))
        print("base_path: ", base_path)
        algo_path = os.path.join(base_path, "models/olympics-running/ppo")
        run_path = os.path.join(algo_path, run_dir)
        run_path = os.path.join(run_path, "trained_model")
        model_actor_path = os.path.join(run_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(run_path, "critic_" + str(episode) + ".pth")
        print(f"Actor path: {model_actor_path}")
        print(f"Critic path: {model_critic_path}")

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=self.device)
            critic = torch.load(model_critic_path, map_location=self.device)
            self.actor_net.load_state_dict(actor)
            self.critic_net.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f"Model not founded!")

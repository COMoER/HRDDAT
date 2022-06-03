import os
import sys
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

father_path = path.dirname(__file__)
sys.path.append(str(os.path.dirname(father_path)))

from rl_trainer.algo.network import Actor, CNN_Actor, CNN_Critic, Critic
from torch.utils.tensorboard import SummaryWriter
from rl_trainer.algo.utils import AvgMeter, RunningMeanStd
from rl_trainer.algo.curiosity import IntrinsicCuriosityModule
import torch.nn.functional as F
import numpy as np


class Args:
    gae_lambda = 0.95
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_update_time = 5
    buffer_capacity = 1500
    batch_size = 32
    gamma = 0.99
    lr = 0.0001

    p_ratio = 1
    v_ratio = 1
    c_ratio = 1
    entropy_beta = 0.001

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
    # use_cnn = args.use_cnn
    v_ratio = args.v_ratio
    p_ratio = args.p_ratio
    c_ratio = args.c_ratio
    entropy_beta = args.entropy_beta

    def __init__(
            self,
            parsed_args,
            run_dir: str = None,
            writer: SummaryWriter = None,
            use_gae: bool = True,
    ):
        super(PPO, self).__init__()

        self.args = args
        device = parsed_args.device
        if parsed_args.use_cnn:
            # print("Using CNN!")
            self.actor_net = CNN_Actor(self.state_space, self.action_space)
            self.critic_net = CNN_Critic(self.state_space)
        else:
            self.actor_net = Actor(self.state_space, self.action_space)
            self.critic_net = Critic(self.state_space, task=parsed_args.task_num, popart=parsed_args.popart,
                                     device=device)

        self.actor_net = self.actor_net.to(device)
        self.critic_net = self.critic_net.to(device)

        # normalize the reward ratio
        self.curiosity = parsed_args.curiosity
        self.ext_ratio = parsed_args.ext_ratio / (parsed_args.ext_ratio + parsed_args.curiosity_ratio + 1e-8)
        self.curiosity_ratio = parsed_args.curiosity_ratio / (
                parsed_args.ext_ratio + parsed_args.curiosity_ratio + 1e-8)

        if self.curiosity:
            self.icm_module = IntrinsicCuriosityModule(args.state_space, num_actions=args.action_space,
                                                       use_random_features=True).to(device)

        self.buffer = []
        self.counter = 0
        self.training_step = 0

        self.params = (
                list(self.actor_net.parameters()) +
                list(self.critic_net.parameters())
        )
        param_lr = [
            {'params': self.actor_net.parameters()},
            {'params': self.critic_net.parameters(), 'lr': 8 * self.lr}
        ]
        if self.curiosity:
            self.params += list(self.icm_module.parameters())
            param_lr += [{'params': self.icm_module.parameters(), 'lr': 5 * self.lr}]

        self.optimizer = optim.Adam(param_lr,
                                    lr=self.lr)

        self.device = device

        self.run_dir = run_dir
        self.writer = writer
        self.IO = True if (run_dir is not None) else False

        self.p_loss = AvgMeter()
        self.v_loss = AvgMeter()
        self.kl_div = AvgMeter()
        self.surprisal = AvgMeter()

        self.use_gae = use_gae
        self.reward_norm = parsed_args.reward_norm
        self.adv_norm = parsed_args.advt_norm

        self.task = parsed_args.task_num if parsed_args.popart else 1
        self.popart = parsed_args.popart

        try:
            self.mini_batch_size = parsed_args.batch_size
        except:
            self.mini_batch_size = 0

        assert not (self.popart and self.reward_norm),"Can't reward norm while using popart!"

        if parsed_args.popart:
            self.reward_tracker = self.critic_net.state_value
        else:
            self.reward_tracker = RunningMeanStd()

    def select_action(self, state, train=True):
        state = torch.from_numpy(state).float().reshape(1, -1).to(self.device)
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

    def update(self, win_is, ep_i):
        """
        Args:
            win_is: bool array presents the winning rollout
        """
        print("start update")
        win_is = win_is.reshape(-1)
        win_idx = list(np.arange(len(win_is))[win_is])
        # # [R,25,25]->[T,R_win,25,25]
        obs = torch.tensor(np.stack([t.state[win_idx] for t in self.buffer]), dtype=torch.float).to(self.device)
        action = torch.tensor(np.stack([t.action[win_idx] for t in self.buffer]), dtype=torch.long) \
            .view(-1, len(win_idx)).to(self.device)
        reward = torch.tensor(np.stack([t.reward[win_idx] for t in self.buffer]), dtype=torch.float). \
            view(-1, len(win_idx)).to(self.device)
        done = torch.tensor(np.stack([t.done[win_idx] for t in self.buffer]), dtype=torch.float) \
            .view(-1, len(win_idx)).to(self.device)


        if self.popart:
            task_mask = torch.tensor(np.stack([t.task_mask[win_idx] for t in self.buffer]), dtype=torch.long) \
                .view(-1, len(win_idx)).to(self.device)-1 # -1 from 1-11 to 0-10
            task_mask = F.one_hot(task_mask,num_classes=self.task)

        old_action_log_prob = torch.tensor(np.stack([t.a_log_prob[win_idx] for t in self.buffer]), dtype=torch.float) \
            .view(-1, len(win_idx)).to(self.device)

        ## Curiosity Inner reward
        if self.curiosity:
            next_obs = torch.empty_like(obs)
            next_obs[:-1, ] = obs[1:, ]
            next_obs[-1,] = obs[0,]  # Back to the start point
            next_feat, next_feat_pred, _ = self.icm_module(x0=obs,
                                                           a=F.one_hot(action, self.action_space),
                                                           x1=next_obs)
            inner_reward = 0.5 * (next_feat - next_feat_pred).pow(2).sum(2)  # shape [T, R]
            if self.IO:
                self.writer.add_scalar("Gt/inner_Gt", torch.sum(inner_reward, 0).mean().item(), ep_i)

            reward = self.ext_ratio * reward + self.curiosity_ratio * inner_reward.detach()

        if self.reward_norm:
            reward_mean, reward_std, count = reward.mean().cpu().detach().numpy(), reward.std().cpu().detach().numpy(), len(
                reward)
            self.reward_tracker.update_from_moments(reward_mean, reward_std ** 2, count)
            std = np.sqrt(self.reward_tracker.var)
            reward /= std

        if self.use_gae:
            # normalized value
            normalized_value = self.critic_net(obs.flatten(2, 3).flatten(0, 1)).detach()\
                .view(-1, len(win_idx),self.task)
            # to be fit the reward scale the return [T,R]
            value = normalized_value.view(-1, len(win_idx)) if not self.popart \
                else self.reward_tracker.denormalize(normalized_value, task_mask, dim=2)
            R = reward[-1] - value[-1]
            Gt = [R]
            mask = (done == 0)
            for step_i in range(len(reward) - 2, -1, -1):
                delta = reward[step_i] + self.gamma * value[step_i + 1] * mask[step_i] - value[step_i]
                R = delta + self.gamma * self.gae_lambda * mask[step_i] * R
                Gt.insert(0, R)
            # T,R_win
            Gt = torch.stack(Gt).float().to(self.device).view(-1, len(win_idx))
            Advt = Gt.clone()
            Gt += value.to(self.device)

            if self.adv_norm:
                Advt = (Advt - torch.mean(Advt, 0, keepdim=True)) / (torch.std(Advt, 0, keepdim=True) + 1e-10)

        # merge rollout and T
        padding_masks = (done != -1).flatten()
        batch_size = len(padding_masks)

        # batch_size = len(win_idx) * len(reward)
        obs = obs.view(batch_size, -1)[padding_masks,]
        if self.curiosity:
            next_obs = next_obs.view(batch_size, -1)[padding_masks,]
        action = action.view(batch_size, 1)[padding_masks,]
        Gt = Gt.view(batch_size)[padding_masks,]
        Advt = Advt.view(batch_size)[padding_masks,]

        if self.popart:
            task_mask = task_mask.view(batch_size, -1)[padding_masks,]

        old_action_log_prob = old_action_log_prob.view(batch_size, 1)[padding_masks,]

        batch_size = padding_masks.sum().item()
        mini_batch_size = int(np.ceil(batch_size / 3)) if self.mini_batch_size <= 0 else self.mini_batch_size
        for i in range(self.ppo_update_time):
            for index in BatchSampler(
                    SubsetRandomSampler(range(batch_size)), mini_batch_size, False
            ):

                Gt_index = Gt[index].view(-1, 1)

                V = self.critic_net(obs[index])

                if self.popart:
                    # update the popart parameter
                    task_mask_index = task_mask[index].view(-1,self.task)
                    self.reward_tracker.update(Gt_index, task_mask_index, dim=1)
                    Gt_index = self.reward_tracker.normalize(Gt_index.view(-1), task_mask_index, dim=1).view(-1,1)
                    V = (V*task_mask_index).sum(dim=-1,keepdim=True)

                # delta = Gt_index - V
                if self.use_gae:
                    advantage = Advt[index]  # no_grad
                # else:
                #     advantage = delta.detach()
                # epoch iteration, PPO core!!!
                all_action_prob = self.actor_net(obs[index])  # [B,A]

                action_prob = all_action_prob.gather(
                    1, action[index]
                )  # new policy

                ratio = action_prob / old_action_log_prob[index]
                surr1 = ratio * advantage.view(-1, 1)
                surr2 = (
                        torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                        * advantage.view(-1, 1)
                )

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)

                entropy = -(action_prob * torch.log(action_prob + 1e-10)).mean()

                loss = PPO.p_ratio * action_loss + PPO.v_ratio * value_loss - PPO.entropy_beta * entropy

                # update Curiosity Pred network
                if self.curiosity:
                    next_feat, next_feat_pred, _ = self.icm_module(x0=obs[index],
                                                                   a=F.one_hot(action[index],
                                                                               self.action_space).squeeze(),
                                                                   x1=next_obs[index])
                    surprisal = F.mse_loss(next_feat, next_feat_pred)
                    loss += PPO.c_ratio * surprisal

                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.params, self.max_grad_norm
                )
                self.optimizer.step()

                self.training_step += 1

                with torch.no_grad():
                    new_action_prob = self.actor_net(obs[index])
                    kl_div = F.kl_div(new_action_prob.log(), all_action_prob, reduction='batchmean')

                self.v_loss += value_loss.item()
                self.p_loss += action_loss.item()
                self.kl_div += kl_div.item()
                if self.curiosity:
                    self.surprisal += surprisal.item()

        if self.IO:
            self.writer.add_scalar("loss/policy loss", float(self.p_loss), ep_i)
            self.writer.add_scalar("loss/critic loss", float(self.v_loss), ep_i)
            self.writer.add_scalar("loss/kl divergence", float(self.kl_div), ep_i)
            if self.curiosity:
                self.writer.add_scalar("loss/suprisal", float(self.surprisal), ep_i)
            if self.popart:
                self.reward_tracker.record(self.writer)

        self.p_loss = AvgMeter()
        self.v_loss = AvgMeter()
        self.kl_div = AvgMeter()
        self.surprisal = AvgMeter()
        self.clear_buffer()

    def clear_buffer(self):
        del self.buffer[:]

    def save(self, save_path, episode):
        base_path = os.path.join(save_path, "trained_model")
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        if self.curiosity:
            ckpt = {
                'actor': self.actor_net.state_dict(),
                'critic': self.critic_net.state_dict(),
                'optim': self.optimizer.state_dict(),
                'icm': self.icm_module.state_dict(),
            }
        else:
            ckpt = {
                'actor': self.actor_net.state_dict(),
                'critic': self.critic_net.state_dict(),
                'optim': self.optimizer.state_dict(),
            }

        if self.curiosity:
            ckpt['icm'] = self.icm_module.state_dict()

        model_path = os.path.join(base_path, f"ckpt_{episode}.pt")
        torch.save(ckpt, model_path)

    def load(self, run_dir, episode, eval=False):
        print(f"\nBegin to load model: ")
        print("run_dir: ", run_dir)
        base_path = os.path.dirname(os.path.dirname(__file__))
        print("base_path: ", base_path)
        algo_path = os.path.join(base_path, "models/olympics-running/ppo")
        run_path = os.path.join(algo_path, run_dir)
        run_path = os.path.join(run_path, "trained_model")
        model_path = os.path.join(run_path, f"ckpt_{episode}.pt")
        print(f"CheckPoint path: {model_path}")
        if os.path.exists(model_path):
            ckpt = torch.load(model_path)
            if eval:
                self.actor_net.load_state_dict(ckpt['actor'])
            else:
                self.actor_net.load_state_dict(ckpt['actor'])
                self.critic_net.load_state_dict(ckpt['critic'])
                self.optimizer.load_state_dict(ckpt['optim'])
                try:
                    self.icm_module.load_state_dict(ckpt['icm'])
                except:
                    if hasattr(self, "icm_module"):
                        sys.exit("Model doesn't have ICM but you need curiosity!")
            print("Model loaded!")
        else:
            sys.exit("Model not founded!")

    def copy_from(self, model):
        self.actor_net.load_state_dict(model.actor_net.state_dict())
        self.critic_net.load_state_dict(model.critic_net.state_dict())
        self.icm_module.load_state_dict(model.icm_module.state_dict())

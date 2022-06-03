"""
Refer to OpenAI Baselines code's environment setting
https://github.com/marlbenchmark/on-policy
"""
from env.olympics_running import OlympicsRunning
from gym.spaces import Box,Discrete
from typing import List
from rl_trainer.algo.utils import DataNormalize

from pathlib import Path
base_dir = str(Path(__file__).resolve().parent.parent.parent)

S = 200
import numpy as np

actions_map = {
    0: [-100, -30],
    1: [-100, -18],
    2: [-100, -6],
    3: [-100, 6],
    4: [-100, 18],
    5: [-100, 30],
    6: [-40, -30],
    7: [-40, -18],
    8: [-40, -6],
    9: [-40, 6],
    10: [-40, 18],
    11: [-40, 30],
    12: [20, -30],
    13: [20, -18],
    14: [20, -6],
    15: [20, 6],
    16: [20, 18],
    17: [20, 30],
    18: [80, -30],
    19: [80, -18],
    20: [80, -6],
    21: [80, 6],
    22: [80, 18],
    23: [80, 30],
    24: [140, -30],
    25: [140, -18],
    26: [140, -6],
    27: [140, 6],
    28: [140, 18],
    29: [140, 30],
    30: [200, -30],
    31: [200, -18],
    32: [200, -6],
    33: [200, 6],
    34: [200, 18],
    35: [200, 30],
}




class OlympicsEnv(object):
    def __init__(self, oly_env: OlympicsRunning, task,shuffle_map=False,data_norm=False,use_astar=True):
        """
        for multi-agent environment
        """
        self.oly_env = oly_env
        self.n_agents = oly_env.n_player
        self.task = task
        self.action_space = [Discrete(len(actions_map.keys())) for _ in range(self.n_agents)]
        self.observation_space = [[25*25] for _ in range(self.n_agents)]
        self.observations = None
        self.rewards = None
        self.dones = np.zeros((self.n_agents,),np.bool_)
        self.share_observation_space = [[2*25*25] for _ in range(self.n_agents)]
        self.episode_limit = 500
        self.shuffle_map = shuffle_map
        self.data_norm = DataNormalize(data_norm)
        self.use_astar = use_astar
        self.last_step_cost = [0.,0.]
        if use_astar:
            self.astar_costmap = \
                np.stack(
                    [np.stack(
                        [np.load(
                            f"{base_dir}/costmap/astar/S{S}/map{map_id}_cost{agent_id}.npy"
                        ) for agent_id in range(2)], axis=0
                    ) for map_id in range(1, 12)], axis=0)

    def _reward_warp(self, reward, done):
        """
        reward: list[int]
        done: bool
        """
        # simple reward shaping
        if not done:
            reward = [-1.0, -1.0]
            if self.use_astar:
                for i in range(self.n_agents):
                    if self.use_astar:
                        current_step_pos = self.oly_env.env_core.agent_pos[i]
                        current_step_pos_index = (
                            int(current_step_pos[1]), int(current_step_pos[0]))

                        # astar_costmap: (map_num=11,agent_num=2,700,700)
                        current_step_cost = float(
                            self.astar_costmap[(self.task - 1, i) + current_step_pos_index])
                        reward_increment = \
                        (current_step_cost - self.last_step_cost[i]) / 5
                        reward[i] += reward_increment

                        self.last_step_cost[i] = current_step_cost

        else:
            if reward[0] != reward[1]:
                if reward[0] < reward[1]:
                    reward = [reward[0] - 100, reward[1]]
                else:
                    reward = [reward[0], reward[1] - 100]
            else:
                reward = [-1.0, -1.0]
        return reward

    def convert_action(self, actions):
        """
        Args:
            actions: (num_agents,2)
        Return:
            formed_actions
        """
        formed_actions = [[[actions[i,0]],
                               [actions[i,1]]]
                              for i in range(self.n_agents)]
        return formed_actions
    def step(self, actions):
        """
        Args:
            actions: np.array([num_agents,2]
        Returns:
            obs[num_agents,25,25],rewards[num_agents,1],dones[num_agents,1], info[num_agents,1]
        """
        infos = np.ones((self.n_agents,),np.int_)*self.oly_env.env_core.map_num
        if not self.dones.all():
            state, rewards, done, _, info = self.oly_env.step(self.convert_action(actions))
            self.rewards = np.array(self._reward_warp(rewards, done)).reshape(-1)
            observations = [self.data_norm.normalize(state[i]['obs']) for i in range(self.n_agents)]
            self.observations = np.stack(observations)
            self.dones = np.array([done for _ in range(self.n_agents)], np.bool_).reshape(-1)
            return self.observations.copy(), self.rewards.copy(), self.dones.copy(), infos
        else:
            dones = np.array([-1, -1]) # Padding Signal
            return self.observations.copy(), self.rewards.copy(), dones, infos


    def reset(self):
        """Returns initial observations and states."""
        state = self.oly_env.reset(shuffle_map=self.shuffle_map)

        observations = [self.data_norm.normalize(state[i]['obs']) for i in range(self.n_agents)]
        self.observations = np.stack(observations)
        self.rewards = None
        self.dones = np.zeros((self.n_agents,),np.bool_)
        return self.observations.copy()
        # return observations

    def render(self):
        pass

    def close(self):
        pass

    def seed(self, seed=None):
        self.oly_env.set_seed(seed)

    def set_max_step(self,max_step):
        self.oly_env.set_max_step(max_step)

    def save_replay(self):
        pass
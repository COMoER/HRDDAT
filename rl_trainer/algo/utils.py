import numpy as np
from pathlib import Path

base_dir = str(Path(__file__).resolve().parent.parent.parent)
import os


class Reward(object):
    S = 200  # pixel
    R_S = 700  # m
    reso = R_S / S

    def __init__(self, map_id=1, num_agent=2):
        self.nagents = num_agent
        self.map_id = 0
        self.reset(map_id)

    def reset(self, map_id):
        if self.map_id != map_id:
            self.map_id = map_id
            self.costmap = np.load(os.path.join(base_dir, "costmap", f"map{self.map_id}.npy"))
        self.agents_energy = [-1 for _ in range(self.nagents)]

    def _pos2grid(self, pos):
        return np.clip(np.round(np.array(pos) / Reward.reso), 0, Reward.S - 1).astype(int)

    def calc_reward(self, agents_pos):
        """
        agents_pos:List([x,y])
        """
        reward = []
        for i, pos in enumerate(agents_pos):
            pos_grid = self._pos2grid(pos)
            if self.agents_energy[i] < 0:
                dE = -self.costmap[pos_grid[1], pos_grid[0]]
            else:
                dE = -self.costmap[pos_grid[1], pos_grid[0]] + self.agents_energy[i]
            self.agents_energy[i] = self.costmap[pos_grid[1], pos_grid[0]]
            reward.append(dE)
        return reward


class AvgMeter(object):
    def __init__(self):
        self.value = 0.
        self.count = 0
    def __add__(self, other:float):
        self.value = (self.value * self.count + other) / (self.count + 1)
        self.count += 1
        return self
    def __float__(self):
        return self.value

class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)


    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class DataNormalize(object):
    def __init__(self,norm):
        self.data = np.load(os.path.join(base_dir,"data_normalize","25x25_data.npz"))
        self.data_channel = np.load(os.path.join(base_dir,"data_normalize","25x25_data_channel.npz"))
        self.mean = self.data['arr_4']
        self.std = self.data['arr_5']
        self.norm = norm
    def normalize(self,obs):
        return (obs-self.mean)/(self.std+1e-10) if self.norm else obs



if __name__ == '__main__':
    a = DataNormalize(True)

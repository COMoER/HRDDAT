from rl_trainer.algo.ppo import PPO
from utils.action_space import actions_map
from rl_trainer.algo.utils import DataNormalize
import torch
from pathlib import Path
import os

base_dir = Path(__file__).parent.parent.parent

####################
run = 2 # choose which run package to evaluate
episode = 1300 # choose which episode check point to evaluate
####################

data_norm_flag = False

agent = PPO()
agent.load(os.path.join(base_dir,"rl_trainer","models","olympics-running","ppo",f"run{run}"),episode)
data_norm = DataNormalize(data_norm_flag)

def get_observation(obs):
    return data_norm.normalize(obs).flatten()
def my_controller(observation, action_space, is_act_continuous=False):
    obs = get_observation(observation["obs"])
    action = agent.choose_action(obs,train=True)
    actions = actions_map[action]
    agent_action = [[actions[0]], [actions[1]]]
    return agent_action
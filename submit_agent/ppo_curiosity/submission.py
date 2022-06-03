from rl_trainer.algo.ppo_parallel_curiosity import PPO
from utils.action_space import actions_map
from rl_trainer.main_parallel_curiosity import get_args
from rl_trainer.algo.utils import DataNormalize
import torch
from pathlib import Path
from collections import namedtuple
import os
import yaml

base_dir = Path(__file__).parent.parent.parent

####################
run = 1 # choose which run package to evaluate
episode = 296 # choose which episode check point to evaluate
####################
with open(os.path.join(Path(__file__).parent.resolve(),"config.yaml"), 'r', encoding="utf-8") as f:
    file_data = f.read()
    default_args = yaml.load(file_data,yaml.SafeLoader)
model_dir = os.path.join(base_dir,"rl_trainer","models","olympics-running","ppo",f"run{run}")
with open(os.path.join(model_dir,"config.yaml"), 'r', encoding="utf-8") as f:
    file_data = f.read()
    args = yaml.load(file_data,yaml.SafeLoader)
    for key,value in default_args.items():
        if key not in args.keys():
            args[key] = value
    Args = namedtuple("args",list(args.keys()))
    parsed_args = Args._make(list(args.values()))
data_norm_flag = parsed_args.data_norm
agent = PPO(parsed_args)
agent.load(model_dir,episode,True)
data_norm = DataNormalize(data_norm_flag)
def get_observation(obs):
    return data_norm.normalize(obs).flatten()
def my_controller(observation, action_space, is_act_continuous=False):
    obs = get_observation(observation["obs"])
    action = agent.choose_action(obs,train=True)
    actions = actions_map[action]
    agent_action = [[actions[0]], [actions[1]]]
    return agent_action
import numpy as np
import torch
import random

from utils.pure_env_wrapper import wrap_pytorch_task

from tabulate import tabulate
import argparse
from torch.distributions import Categorical
import os
from tqdm import tqdm
import importlib

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_actions(state, algo, ctrl_agent,action_space):

    actions = algo.my_controller(state[ctrl_agent],action_space[ctrl_agent])
    return actions


def get_join_actions(state, agent_list,joint_action_space):

    joint_actions = []

    for agent_idx in range(len(agent_list)):
        joint_actions.append(get_actions(state,agent_list[agent_idx],agent_idx,joint_action_space))
    return joint_actions


def run_game(env, algo_list, algo_name, episode, shuffle_map,render):
    total_reward = np.zeros(2)
    num_win = np.zeros(3)  # agent 1 win, agent 2 win, draw
    episode = int(episode)
    for i in tqdm(range(1, int(episode) + 1)):
        episode_reward = np.zeros(2)

        state = env.reset(shuffle_map)

        for a in algo_list:
            try:
                a.reset()
            except:
                # print(a.__name__)
                pass
        if render:
            env.env_core.render()

        step = 0

        while True:
            joint_action = get_join_actions(state, agent_list,env.joint_action_space)
            next_state, reward, done, _, info = env.step(joint_action)
            reward = np.array(reward)
            episode_reward += reward
            if render:
                env.env_core.render()

            if done:
                if reward[0] != reward[1]:
                    if reward[0] == 100:
                        num_win[0] += 1
                    elif reward[1] == 100:
                        num_win[1] += 1
                    else:
                        raise NotImplementedError
                else:
                    num_win[2] += 1

                break
            state = next_state
            step += 1
        total_reward += episode_reward
    total_reward /= episode
    print("total reward: ", total_reward)
    # print("Result in map {} within {} episode:".format(map_num, episode))

    header = ["Name", algo_name[0], algo_name[1]]
    data = [
        ["score", np.round(total_reward[0], 2), np.round(total_reward[1], 2)],
        ["win", num_win[0], num_win[1]],
    ]
    print(tabulate(data, headers=header, tablefmt="pretty"))



if __name__ == "__main__":
    game = wrap_pytorch_task("olympics-running:1")


    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default="ppo")
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--episode", default=100)
    parser.add_argument("--map_num",type=int,default=1)
    parser.add_argument("--shuffle_map", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    my_agent = importlib.import_module("submit_agent.{}.submission".format(args.my_ai))
    oppo_agent = importlib.import_module("submit_agent.{}.submission".format(args.opponent))

    agent_list = [my_agent,oppo_agent]
    # agent_list = [args.my_ai,args.opponent]
    game.specify_a_map(args.map_num)
    game.set_max_step(400)
    run_game(game, algo_list=agent_list,algo_name = [args.my_ai,args.opponent],episode=args.episode,shuffle_map=args.shuffle_map,
             render=args.render)

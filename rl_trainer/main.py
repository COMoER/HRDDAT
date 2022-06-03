import argparse
import datetime
import random
import sys
from pathlib import Path
from pprint import pprint
from typing import Dict

import numpy as np
import torch
import time
from torch.utils.tensorboard import SummaryWriter

base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)

from collections import deque, namedtuple

from env.chooseenv import make

from rl_trainer.algo.ppo import PPO
from rl_trainer.algo.random_agent import random_agent
from rl_trainer.log_path import *
from rl_trainer.algo.prof import Timings
from rl_trainer.algo.utils import DataNormalize

import logging

from utils.action_space import actions_map

algo_name_list = ["ppo"]
algo_list = [PPO]
algo_map = dict(zip(algo_name_list, algo_list))


def get_game(seed: int = None, config: Dict = None):
    return make("olympics-running", seed, config)


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_name", default="olympics-running", type=str)
    parser.add_argument(
        "--algo",
        default="ppo",
        type=str,
        help="the algorithm to use",
        choices=algo_name_list,
    )

    parser.add_argument("--max_episodes", default=1500, type=int)
    parser.add_argument("--episode_length", default=500, type=int)
    parser.add_argument(
        "--map", default=1, type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    )
    parser.add_argument("--shuffle_map", action="store_true")
    parser.add_argument("--shuffle_index", action="store_true")

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "cuda"])
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--load_run", default=1, type=int)
    parser.add_argument("--load_episode", default=500, type=int)

    parser.add_argument("--log", action="store_false", default=True)

    parser.add_argument("--reward_norm", default=False, action="store_true")
    parser.add_argument("--data_norm", default=False, action="store_true")
    parser.add_argument("--advt_norm", default=False, action="store_true")

    parser.add_argument("--ext_ratio", type=float, default=0.5)
    parser.add_argument("--curiosity_ratio", type=float, default=0.5)
    return parser.parse_args()


def main(args):
    pprint(args.__dict__)

    env = get_game(args.seed)
    env.set_max_step(500)
    if not args.shuffle_map:
        env.specify_a_map(
            args.map
        )  # specifying a map, you can also shuffle the map by not doing this step
    env.set_seed(0)

    num_agents = env.n_player
    print(f"Total agent number: {num_agents}")

    ctrl_agent_index = 1
    print(f"Agent control by the actor: {ctrl_agent_index}")

    width = env.env_core.view_setting["width"] + 2 * env.env_core.view_setting["edge"]
    height = env.env_core.view_setting["height"] + 2 * env.env_core.view_setting["edge"]
    print(f"Game board width: {width}")
    print(f"Game board height: {height}")

    act_dim = env.action_dim
    obs_dim = 25 * 25
    print(f"action dimension: {act_dim}")
    print(f"observation dimension: {obs_dim}")

    data_norm = DataNormalize(args.data_norm)

    sys.stdout.flush()

    setup_seed(args.seed)
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    if args.log:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        log_fname = os.path.join(log_dir, 'log_train.txt')
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
        logger = logging.getLogger("Trainer")

        print(f"store in {run_dir}")

        if not args.load_model:
            writer = SummaryWriter(
                os.path.join(
                    str(log_dir),
                    "{}_{} on map {}".format(
                        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                        args.algo,
                        "all" if args.shuffle_map else args.map,
                    ),
                )
            )
            save_config(args, log_dir)
    prof = Timings()
    record_win = deque(maxlen=100)
    record_win_op = deque(maxlen=100)

    algo = algo_map[args.algo]

    if args.load_model:
        model = algo(args.device)
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_run))
        model.load(load_dir, episode=args.load_episode)
    else:
        if args.log:
            model = PPO(args.device, run_dir, writer)
        else:
            model = PPO(args.device)
    Transition = namedtuple(
        "Transition",
        ["state", "action", "a_log_prob", "reward", "next_state", "done"],
    )

    opponent_agent = random_agent()  # we use random opponent agent here

    episode = 0
    train_count = 0

    while episode < args.max_episodes:
        state = env.reset(args.shuffle_map)

        # shuffle index
        if args.shuffle_index:
            ctrl_agent_index = random.randint(0, 1)

        if args.render:
            env.env_core.render()
        obs_ctrl_agent = data_norm.normalize(np.array(state[ctrl_agent_index]["obs"])).flatten()
        obs_oppo_agent = data_norm.normalize(state[1 - ctrl_agent_index]["obs"])

        episode += 1
        step = 0
        Gt = 0

        while True:
            prof.reset()
            action_opponent = opponent_agent.choose_action(
                obs_oppo_agent
            )  # opponent action
            # action_opponent = [
            #     [0],
            #     [0],
            # ]  # here we assume the opponent is not moving in the demo

            action_ctrl_raw, action_prob = model.select_action(
                obs_ctrl_agent, False if args.load_model else True
            )
            # inference
            action_ctrl = actions_map[action_ctrl_raw]
            action_ctrl = [[action_ctrl[0]], [action_ctrl[1]]]  # wrapping up the action

            action = (
                [action_opponent, action_ctrl]
                if ctrl_agent_index == 1
                else [action_ctrl, action_opponent]
            )
            prof.time("step")
            next_state, reward, done, _, info = env.step(action)
            prof.time("env")

            next_obs_ctrl_agent = data_norm.normalize(next_state[ctrl_agent_index]["obs"])
            next_obs_oppo_agent = data_norm.normalize(next_state[1 - ctrl_agent_index]["obs"])

            step += 1

            # simple reward shaping
            if not done:
                ext_reward = [-1.0, -1.0]
            else:
                if reward[0] != reward[1]:
                    ext_reward = (
                        [reward[0] - 100, reward[1]]
                        if reward[0] < reward[1]
                        else [reward[0], reward[1] - 100]
                    )
                else:
                    ext_reward = [-1.0, -1.0]

            reward = [ext_reward[0], ext_reward[1]]

            trans = Transition(
                obs_ctrl_agent,
                action_ctrl_raw,
                action_prob,
                reward[ctrl_agent_index],
                next_obs_ctrl_agent,
                done,
            )
            model.store_transition(trans)

            obs_oppo_agent = next_obs_oppo_agent
            obs_ctrl_agent = np.array(next_obs_ctrl_agent).flatten()
            if args.render:
                env.env_core.render()
            Gt += ext_reward[ctrl_agent_index] if done else -1
            prof.time("push")
            if done:
                win_is = (
                    1 if ext_reward[ctrl_agent_index] > ext_reward[1 - ctrl_agent_index] else 0
                )
                win_is_op = (
                    1 if ext_reward[ctrl_agent_index] < ext_reward[1 - ctrl_agent_index] else 0
                )
                record_win.append(win_is)
                record_win_op.append(win_is_op)
                print(
                    "Episode: ",
                    episode,
                    "controlled agent: ",
                    ctrl_agent_index,
                    "; Episode Return: ",
                    Gt,
                    "; win rate(controlled & opponent): ",
                    "%.2f" % (sum(record_win) / len(record_win)),
                    "%.2f" % (sum(record_win_op) / len(record_win_op)),
                    "; Trained episode:",
                    train_count,
                )
                print(prof.summary())
                sys.stdout.flush()
                if args.log:
                    logger.info(
                        f"\nEpisode: {episode}; controlled agent: {ctrl_agent_index}" +
                        f"; Episode Return: {Gt}" +
                        "; win rate(controlled & opponent): " +
                        f"{sum(record_win) / len(record_win):.2f} {sum(record_win_op) / len(record_win_op):.2f}" +
                        f"; Trained episode: {train_count}"
                    )
                    logger.info(prof.summary())

                if args.algo == "ppo" and len(model.buffer) >= model.batch_size:
                    if win_is == 1:
                        model.update(episode)
                        train_count += 1
                    else:
                        model.clear_buffer()

                if args.log:
                    writer.add_scalar("training Gt", Gt, episode)
                    writer.add_scalar("win_rate",sum(record_win) / len(record_win),episode)

                break
        if episode % args.save_interval == 0 and not args.load_model and args.log:
            model.save(run_dir, episode)
            writer.close()


if __name__ == "__main__":
    args = get_args()
    main(args)

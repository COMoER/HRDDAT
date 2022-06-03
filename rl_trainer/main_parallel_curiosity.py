import argparse
import datetime
import random
import sys
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)

from collections import deque, namedtuple

from rl_trainer.algo.ppo_parallel_curiosity import PPO
from rl_trainer.algo.random_agent import random_agent
from rl_trainer.log_path import *
from rl_trainer.algo.prof import Timings
from rl_trainer.envs.env_wrappers import wrap_pytorch_task

import logging

from utils.action_space import actions_map

algo_name_list = ["ppo"]
algo_list = [PPO]
algo_map = dict(zip(algo_name_list, algo_list))


# def get_game(seed: int = None, config: Dict = None):
#     return make("olympics-running", seed, config)


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

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--device", default="cpu", type=str, choices=["cpu", "cuda"])
    parser.add_argument("--save_interval", default=10, type=int)
    parser.add_argument("--render", action="store_true")

    parser.add_argument("--load_model", action="store_true")
    parser.add_argument("--load_run", default=1, type=int)
    parser.add_argument("--load_episode", default=300, type=int)
    parser.add_argument("--curiosity", default=False, action="store_true")
    parser.add_argument("--all_train", default=False, action="store_true")

    parser.add_argument("--reward_norm", default=False, action="store_true")
    parser.add_argument("--data_norm", default=False, action="store_true")
    parser.add_argument("--advt_norm", default=False, action="store_true")
    parser.add_argument("--popart",default=False,action="store_true")

    parser.add_argument("--ext_ratio", type=float, default=0.5)
    parser.add_argument("--curiosity_ratio", type=float, default=0.5)

    parser.add_argument("--num_rollouts", default=22, type=int)
    parser.add_argument("--log", action="store_false", default=True)
    parser.add_argument("--max_length",default=500,type=int)
    parser.add_argument("--task_num",default=11,type=int)
    parser.add_argument("--batch_size",default=0,type=int,help="divided into 3 mini batch[default], "
                                                               "else fix size mini batch")
    
    parser.add_argument("--use_cnn",default=False,action="store_true")
    parser.add_argument("--use_onehot",default=False,action='store_true')

    parser.add_argument("--use_astar",default=False,action="store_true")
    return parser.parse_args()


def main(args):
    pprint(args.__dict__)

    rollout_maps = [args.map for _ in range(args.num_rollouts)]
    # env = get_game(args.seed)
    env = wrap_pytorch_task(args.game_name,
                            num_rollouts=args.num_rollouts,
                            seed=args.seed,
                            shuffle_map=args.shuffle_map,
                            rollout_maps=rollout_maps,
                            data_norm=args.data_norm,
                            use_astar=args.use_astar)
    setup_seed(args.seed)

    for i in range(args.num_rollouts):
        env.set_max_step(args.max_length, i)

    # default rollouts policy, half control the first and the other half control the second
    half_rollouts = (args.num_rollouts+1) // 2 # ceil instead of floor so there would not be 2 in ctrl_agent_idx
    ctrl_agent_idx = [i // half_rollouts for i in range(args.num_rollouts)]

    prof = Timings()
    run_dir, log_dir = make_logpath(args.game_name, args.algo)

    # normalize the reward ratio
    # ext_ratio = args.ext_ratio / (args.ext_ratio + args.curiosity_ratio + 1e-8)
    # curiosity_ratio = args.curiosity_ratio / (args.ext_ratio + args.curiosity_ratio + 1e-8)


    if args.log:
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        log_fname = os.path.join(log_dir, 'log_train.txt')
        LOGGING_FORMAT = '%(asctime)s %(levelname)s: %(message)s'
        DATE_FORMAT = '%Y%m%d %H:%M:%S'
        logging.basicConfig(level=logging.DEBUG, format=LOGGING_FORMAT, datefmt=DATE_FORMAT, filename=log_fname)
        logger = logging.getLogger("Trainer")

        print(f"store in {run_dir}")

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

    record_win = deque(maxlen=100 * args.num_rollouts)
    record_win_op = deque(maxlen=100 * args.num_rollouts)

    algo = algo_map[args.algo]

    if args.load_model:
        if args.log:
            model = algo(args, run_dir, writer)
        else:
            model = algo(args)
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_run))
        model.load(load_dir, episode=args.load_episode)
    else:
        if args.log:
            model = PPO(args, run_dir, writer)
        else:
            model = PPO(args)

    Transition = namedtuple(
        "Transition",
        ["state", "action", "a_log_prob", "reward", "next_state", "task_mask","done"],
    )

    opponent_agent = random_agent()  # we use random opponent agent here

    episode = 0
    train_count = 0
    while episode < args.max_episodes:
        obs = env.reset()  # [R,N,25,25]

        episode += 1
        step = 0
        Gt = np.zeros((args.num_rollouts,))

        while True:
            prof.reset()
            # action selection
            actions_rollouts = []
            actions_ctrl_raw_rollouts = []
            actions_prob_rollouts = []
            for r, ctrl_idx in enumerate(ctrl_agent_idx):
                actions = [None, None]

                action_ctrl_raw, action_prob = model.select_action(obs[r, ctrl_idx])

                actions[ctrl_idx] = np.array(actions_map[action_ctrl_raw])

                actions[1 - ctrl_idx] = np.array(opponent_agent.choose_action(obs[r, 1 - ctrl_idx])).reshape(-1)

                actions_ctrl_raw_rollouts.append(action_ctrl_raw)
                actions_prob_rollouts.append(action_prob)
                actions_rollouts.append(np.stack(actions))

            actions_ctrl_raw_rollouts = np.array(actions_ctrl_raw_rollouts)
            actions_prob_rollouts = np.array(actions_prob_rollouts)

            prof.time("step")

            # [R,N,25,25],[R,N,2,1] [R,N,2,1],[R,N]
            next_obs, reward, done, task = env.step(actions_rollouts)
            prof.time("env")

            ctrl_agent_idx_shift = [ctrl_idx + i * 2 for i, ctrl_idx in enumerate(ctrl_agent_idx)]


            ext_reward_ctrl = reward.reshape(-1)[ctrl_agent_idx_shift].reshape(args.num_rollouts)
            task_ctrl = task.reshape(-1)[ctrl_agent_idx_shift].reshape(args.num_rollouts)
            obs_ctrl = obs.reshape(-1, 25, 25)[ctrl_agent_idx_shift].reshape(args.num_rollouts, 25, 25)
            next_obs_ctrl = next_obs.reshape(-1, 25, 25)[ctrl_agent_idx_shift].reshape(args.num_rollouts, 25, 25)

            done_ctrl = done.reshape(-1)[ctrl_agent_idx_shift].reshape(args.num_rollouts)
            step += 1

            trans = Transition(
                obs_ctrl,
                actions_ctrl_raw_rollouts,
                actions_prob_rollouts,
                ext_reward_ctrl,
                next_obs_ctrl,
                task_ctrl,
                done_ctrl,
            )
            model.store_transition(trans)

            obs = next_obs
            # if args.render:
            #     env.env_core.render()

            Gt += ext_reward_ctrl * (done_ctrl == 0)
            prof.time("push")

            if (done != 0).all():
                op_agent_idx_shift = [(1 - ctrl_idx) + i * 2 for i, ctrl_idx in enumerate(ctrl_agent_idx)]
                reward_op = reward.reshape(-1)[op_agent_idx_shift].reshape(args.num_rollouts)
                win_is = ext_reward_ctrl > reward_op
                win_is_op = ext_reward_ctrl < reward_op

                Gt += ext_reward_ctrl

                record_win.extend(list(win_is))
                record_win_op.extend(list(win_is_op))

                # if not args.load_model:
                if args.algo == "ppo" and len(model.buffer) >= model.batch_size:
                    if args.curiosity or args.all_train:
                        model.update(np.ones_like(win_is), episode)
                        train_count += np.sum(np.ones_like(win_is))
                    else:
                        if win_is.any():
                            model.update(win_is, episode)
                            train_count += np.sum(win_is)
                        else:
                            model.clear_buffer()


                if args.log:
                    logger.info(
                        f"\nEpisode: {episode};Rollouts: {args.num_rollouts}; Episode Mean Return: {Gt.mean()}" +
                        "; win rate(controlled & opponent): " +
                        f"{sum(record_win) / len(record_win):.2f} {sum(record_win_op) / len(record_win_op):.2f}" +
                        f"; Trained episode: {train_count}"
                    )
                    logger.info('\n' + prof.summary())
                print(f"\nEpisode: {episode};Rollouts: {args.num_rollouts}; Episode Mean Return: {Gt.mean()}" +
                      "; win rate(controlled & opponent): " +
                      f"{sum(record_win) / len(record_win):.2f} {sum(record_win_op) / len(record_win_op):.2f}" +
                      f"; Trained episode: {train_count}" + '\n' + prof.summary())
                if args.log:
                    writer.add_scalar("Gt/Gt_mean", Gt.mean(), episode)
                    for r in range(args.num_rollouts):
                        writer.add_scalar(f"Gt/Gt_{r}",Gt[r],episode)
                    writer.add_scalar("win rate", sum(record_win) / len(record_win), episode)
                sys.stdout.flush()
                break
        if episode % args.save_interval == 0 and not args.load_model and args.log:
            model.save(run_dir, episode)
        if args.log:
            writer.flush()
    if args.log:
        writer.close()
    env.close()


if __name__ == "__main__":
    args = get_args()
    main(args)

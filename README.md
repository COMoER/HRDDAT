## Heuristic Reward Driven Athlete Trainer

Modified from [https://github.com/sjtu-marl/Competition_Olympics-Running](https://github.com/sjtu-marl/Competition_Olympics-Running).
- add evaluate_local script to accept submission.py evaluation
- HRDDAT model train script added (main_parallel_curiosity.py)

### Usage
- install

```shell
pip install -r requirement.txt
# pre generate costmap for astar reward
python astarmap.py
```
- training
```shell
# training baseline with map1
python rl_trainer/main.py --device cuda --map 1
# training baseline with shuffle map
python rl_trainer/main.py --device cuda --shuffle_map
# training HRDDAT best model (we trained on the 64 cpu server)
python rl_trainer/main_parallel_curiosity.py --device cpu --reward_norm --data_norm --advt_norm --curiosity --num_rollouts 36 --max_length 500 --shuffle_map --ext_ratio 0.2 --curiosity_ratio 0.8
```

- evaluation
```shell
# evaluate the model with random opponent
python evaluation_local.py --my_ai ppo_curiosity --opponent random --shuffle_map
# evaluate the model with jidi rl opponent
python evaluation.py --my_ai ppo_curiosity  --opponent rl --shuffle_map
# evaluate the baseline model(trained by main.py) with random opponent
python evaluation_local.py --my_ai ppo --opponent random --shuffle_map
```
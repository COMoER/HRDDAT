import random
class random_agent:
    def __init__(self, seed=None):
        self.force_range = [-100, 200]
        self.angle_range = [-30, 30]
        self.seed(seed)

    def seed(self, seed=None):
        random.seed(seed)

    def choose_action(self, obs):
        force = random.uniform(self.force_range[0], self.force_range[1])
        angle = random.uniform(self.angle_range[0], self.angle_range[1])

        return [[force], [angle]]

agent = random_agent()
agent.seed(0)

def my_controller(observation, action_space, is_act_continuous=False):
    agent_action = agent.choose_action(None)
    return agent_action
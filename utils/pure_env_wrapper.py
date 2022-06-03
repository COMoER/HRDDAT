import env.chooseenv as ech
from env.olympics_running import OlympicsRunning
def wrap_pytorch_task(env_setting: str,ctrl_agent_idx = 0):
    """
    env : “{env_name}:{map_index}"
    """
    try:
        env_name, map_index = env_setting.split(':')
        environ: OlympicsRunning = ech.make(env_name)
        map_index = int(map_index)
        assert map_index in [1,2,3,4,5,6,7,8,9,10,11]
        environ.specify_a_map(map_index)
        return environ
    except:
        raise NotImplementedError
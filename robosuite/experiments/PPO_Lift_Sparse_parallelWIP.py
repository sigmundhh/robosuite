from unicodedata import name
import robosuite as suite
from robosuite.wrappers import GymWrapper
import stable_baselines3 
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import gym
import os
import time
import multiprocessing

models_dir = f"models/PPO-{int(time.time())}"
logdir = f"logs/PPO-{int(time.time())}"



# Define function that returns a SubprocVectorEnv of the given GymWrapped env
def make_vec_env(env_params, n_envs=8):
    """
    Create a wrapped, vectorized gym environment.
    :param env: (GymWrapper) the Gym-wrapped Robosuite environment
    :param n_envs: (int) the number of environments you wish to have in parallel
    :return: (SubprocVecEnv) the vectorized environment
    """
    # Create initialization function for the environment
    env_init = lambda : GymWrapper(suite.make(**env_params))
    env_inits = [env_init for _ in range(n_envs)]

    return SubprocVecEnv(env_inits)


def make_robosuite_env(options):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param options: (dict) additional arguments to pass to the specific environment class initializer
    """
    def _init():
        env = GymWrapper(suite.make(**options))
        env = Monitor(env)
        return env
    return _init

    
if __name__ ==  '__main__':
    #Create the models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    #Create the logs directory if it doesn't exist
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Make Lift env
    # Problem, I need to pase the initializer to SubprocVecenv, so here I just need to specify the params
    env_params = {
            "env_name" : "Lift",
            "robots" : "IIWA",
            "has_offscreen_renderer" : False,
            "use_camera_obs" : False,
            "use_object_obs" : True,
            "reward_shaping" : True
    }
    ## Check available cpus
    #cpu_count = multiprocessing.cpu_count() # Gives 8 on local
    cpu_count = 8 # Gives 8 on local
    ## Vectorize env over available cpus
    #env = make_vec_env(env_params, n_envs=cpu_count)
    # Create vectorized envs
    #env = SubprocVecEnv([lambda : Monitor(GymWrapper(suite.make(**env_params))) for _ in range(cpu_count)])
    env = SubprocVecEnv([make_robosuite_env(env_params) for i in range(cpu_count)])
    env = VecNormalize(env)

    # Initialize policy
    model = stable_baselines3.PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

    TIMESTEPS = 10000
    for i in range(1,100):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/{TIMESTEPS*i}")


    env.close()
from distutils.command.config import config
from typing import Callable
from unicodedata import name
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers import GymWrapperRGBD
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from wandb.integration.sb3 import WandbCallback
import wandb
import numpy as np
import gym
import os
import time
import multiprocessing
import argparse

config = {
    "env_params": {
        "env_name" : "Lift",
        "robots" : "IIWA",
        "has_renderer" : False,
        "has_offscreen_renderer" : False,
        "use_object_obs" : True,
        "use_camera_obs" : False,
        "reward_shaping" : True,
        "controller_configs" : suite.load_controller_config(
            default_controller="OSC_POSE")
    },
    "total_timesteps": int(1e6),
    "timesteps_pr_save": int(1e5),
    "algorithm" : "PPO",
    "policy_model" : "MlpPolicy",
    "num_processes" : multiprocessing.cpu_count(),
    "random_seed" : 42
}

run = wandb.init(
    project="robosuite_lift_dense_object_obs",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional, what does this imply?
    mode="disabled" # for test-runds
)


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value = np.random.random()
        self.logger.record('random_value', value)
        return True


def make_env(env_params: dict, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = Monitor(GymWrapper(suite.make(**env_params)))
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=f'Train a {config["algorithm"]} agent on the Lift environment')
    parser.add_argument('--cont', type=str, required=False,
                    help='Instance id of the model to continue training')
    return parser.parse_args() 

    
if __name__ ==  '__main__':

    # Parse arguments
    args = parse_arguments()
    reward_type = "dense" if config["env_params"]["reward_shaping"] else "sparse"

    # The folders for weights and logs and their filenames
    run_name = f'{config["algorithm"]}-{config["env_params"]["env_name"]}-{reward_type}-'
    models_dir = f'./models/{run_name}'
    logdir = f'./logs/{run_name}'
    video_dir = f'./videos/{run_name}'

    # Make Lift environment
    env = SubprocVecEnv([make_env(config["env_params"], i, config["random_seed"]) 
        for i in range(config["num_processes"])])

    #env = VecVideoRecorder(env, video_dir, record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
    

    #Check if continue training argument is given
    if args.cont != None: #Continue previous model
        
        instance_id = args.cont

        #Check that model and logs exist for given instance id
        if not os.path.exists(models_dir+instance_id) or not os.path.exists(logdir+instance_id):
            raise ValueError(f"No model or log found for instance id {instance_id}")
        
        model = sb3.PPO.load(models_dir+instance_id, env)
    
    else:   # We want to make new instance
        instance_id = str(int(time.time()))
        
        #Check that instance does not already exist (if several computers train in parallel)
        if os.path.exists(models_dir+instance_id) or os.path.exists(logdir+instance_id):
            raise ValueError(f"Model or log already exists for instance id {instance_id}")
        
        #Create the model and log folders
        os.makedirs(models_dir+instance_id)
        os.makedirs(logdir+instance_id)
        
        # Initialize policy
        model = sb3.PPO(config["policy_model"], env, verbose=1, tensorboard_log=logdir+instance_id)
    
    # Train the model
    training_iterations = config["total_timesteps"] // config["timesteps_pr_save"]
    learning_timesteps = config["timesteps_pr_save"]
    
    try:
        for i in range(training_iterations):
            model.learn(total_timesteps=learning_timesteps, reset_num_timesteps=False, callback=WandbCallback())#, callback=TensorboardCallback)
            model.save(f"{models_dir+instance_id}/{learning_timesteps*i}")
        run.finish()
    except KeyboardInterrupt:
        env.close()
        run.finish()
        print("Closed environment")
    
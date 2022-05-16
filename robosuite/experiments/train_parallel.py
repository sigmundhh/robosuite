from distutils.command.config import config
from typing import Callable
from unicodedata import name
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers import GymWrapperRGBD
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
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
    "reward_type": "dense",
    "observation_type": "rgb_depth",
    "algorithm" : "SAC",
    "total_timesteps": int(2e6),
    "timesteps_pr_save": int(1e5),
    "num_processes" : 8,
    "random_seed" : 42
}


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


def make_env(env_params: dict, rank: int = 0, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    Supports both RGB-D images or flattened observations 
    In the case of "use_camera_obs" = True, a GymWrapperRGBD is used.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        if env_params["use_camera_obs"]:
            env = Monitor(GymWrapperRGBD(suite.make(**env_params), keys=['agentview_image', 'agentview_depth']))
        else:
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

    ## Parameters common to all scenarios
    env_params = {
        "env_name" : "Lift",
        "robots" : "IIWA",
        #"has_renderer" : False,
        #"has_offscreen_renderer" : True,
        #"use_object_obs" : True,
        #"use_camera_obs" : True,
        #"reward_shaping" : True,
        "controller_configs" : suite.load_controller_config(
            default_controller="OSC_POSE"),
        "horizon" : 200,
    }

    # Set observation type
    if config["observation_type"] == "rgb_depth":
        env_params.update({"has_offscreen_renderer" : True, "use_object_obs" : False, "use_camera_obs" : True, "camera_depths" : True})
        policy_model = "CnnPolicy"
    elif config["observation_type"] == "object_obs":
        env_params.update({"has_offscreen_renderer" : False, "use_object_obs" : True, "use_camera_obs" : False})
        policy_model = "MlpPolicy"
    else:
        raise ValueError("Unknown observation type")

    # Set reward type
    if config["reward_type"] == "dense":
        env_params.update({"reward_shaping" : True})
    elif config["reward_type"] == "sparse":
        env_params.update({"reward_shaping" : False})
    else:
        raise ValueError("Unknown reward type")

    run = wandb.init(
        project="robosuite_lift_dense_object_obs",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        #monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional, what does this imply?
        #monitor_gym=True,
        mode="disabled" # for testing
        )

    # Parse arguments
    args = parse_arguments()
    reward_type = config["reward_type"]

    # The folders for weights and logs and their filenames
    run_name = f'{config["algorithm"]}-{env_params["env_name"]}-{reward_type}-'
    models_dir = f'./models/{run_name}'
    logdir = f'./logs/{run_name}'
    video_dir = f'./videos/{run_name}'

    # Make vectorized Lift environment
    env = DummyVecEnv([make_env(env_params, i, config["random_seed"]) 
        for i in range(config["num_processes"])])

    

    # Evaluation environment
    eval_env_config = env_params.copy()
    eval_env_config.update({"reward_shaping" :  False, "has_offscreen_renderer": True})   # Sparse rewards and renderer for evaluation
    eval_env = DummyVecEnv([make_env(eval_env_config)])
    eval_callback = EvalCallback(eval_env, eval_freq=500, 
                             deterministic=True) # Check: Do I need to pass the env into this another way?
    # VIedo recording of eval env
    env = VecVideoRecorder(eval_env, f"videos/{run.id}", record_video_trigger=lambda x: x % 100000 == 0, video_length=200)

    #Check if continue training argument is given
    if args.cont != None: #Continue previous model
        instance_id = args.cont
        #Check that model and logs exist for given instance id
        if not os.path.exists(models_dir+instance_id) or not os.path.exists(logdir+instance_id):
            raise ValueError(f"No model or log found for instance id {instance_id}")
        if config["algorithm"] == "PPO":
            model = sb3.PPO.load(models_dir+instance_id, env)
        elif config["algorithm"] == "SAC":
            model = sb3.SAC.load(models_dir+instance_id, env)
    
    else:   # We want to make new model instance
        instance_id = str(int(time.time()))
        #Check that instance does not already exist (if several computers train in parallel)
        if os.path.exists(models_dir+instance_id) or os.path.exists(logdir+instance_id):
            raise ValueError(f"Model or log already exists for instance id {instance_id}")
        
        #Create the model and log folders
        os.makedirs(models_dir+instance_id)
        os.makedirs(logdir+instance_id)
        
        # Initialize policy
        if config["algorithm"] == "SAC":
            model = sb3.SAC(policy_model, env, verbose=1, tensorboard_log=logdir+instance_id, seed=config["random_seed"])
        elif config["algorithm"] == "PPO":
            model = sb3.PPO(policy_model, env, verbose=1, tensorboard_log=logdir+instance_id, seed=config["random_seed"])
    
    # Train the model
    training_iterations = config["total_timesteps"] // config["timesteps_pr_save"]
    learning_timesteps = config["timesteps_pr_save"]
    
    try:
        for i in range(training_iterations):
            model.learn(total_timesteps=learning_timesteps, reset_num_timesteps=False, callback=[WandbCallback(), eval_callback])
            model.save(f"{models_dir+instance_id}/{learning_timesteps*(i+1)}")
        env.close()
        run.finish()
        print("Run successfully finished. Closed environment")
    except KeyboardInterrupt:
        env.close()
        run.finish()
        print("Closed environment")
    

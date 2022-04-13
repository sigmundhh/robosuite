from distutils.command.config import config
from unicodedata import name
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers import GymWrapperRGBD
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
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
        "has_offscreen_renderer" : True,
        "use_object_obs" : False,
        "use_camera_obs" : True,
        "reward_shaping" : False,
        "camera_depths" : True,
        "camera_heights" : 128,
        "camera_widths" : 128,
        "controller_configs" : suite.load_controller_config(
            default_controller="JOINT_POSITION")
    },
    "observations": ['agentview_image', 'agentview_depth'],
    "total_timesteps": int(1e6),
    "timesteps_pr_save": int(1e4),
    "algorithm" : "PPO",
}

run = wandb.init(
    project="sb3",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional, what does this imply?
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

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=f'Train a {config["algorithm"]} agent on the Lift environment')
    parser.add_argument('--cont', type=str, required=False,
                    help='Instance id of the model to continue training')
    return parser.parse_args() 

    
if __name__ ==  '__main__':
    # Make Lift environment
    env = DummyVecEnv([lambda : Monitor(GymWrapperRGBD(suite.make(**config["env_params"]), 
        keys=config["observations"]))])
    # Parse arguments
    args = parse_arguments()
    reward_type = "dense" if config["env_params"]["reward_shaping"] else "sparse"
    # The folders for weights and logs and their filenames
    models_dir = f'./models/{config["algorithm"]}-{config["env_params"]["env_name"]}-{reward_type}-'
    logdir = f'./logs/{config["algorithm"]}-{config["env_params"]["env_name"]}-{reward_type}-'
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
        model = sb3.PPO("CnnPolicy", env, verbose=1, tensorboard_log=logdir+instance_id)
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
    
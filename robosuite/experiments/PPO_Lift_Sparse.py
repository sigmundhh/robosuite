from unicodedata import name
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers import GymWrapperRGBD
import stable_baselines3 
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import gym
import os
import time
import multiprocessing
import argparse

instance_id = int(time.time())
models_dir = f"./models/PPO-{instance_id}"
logdir = f"./logs/PPO-{instance_id}"



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

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train a PPO agent on the Lift environment with sparse rewards')
    parser.add_argument('--cont', type=str, required=False,
                    help='Instance id of the model to continue training')
    return parser.parse_args() 

    
if __name__ ==  '__main__':
    # Make Lift env
    # Problem, I need to pase the initializer to SubprocVecenv, so here I just need to specify the params
    env_params = {
            "env_name" : "Lift",
            "robots" : "IIWA",
            "has_offscreen_renderer" : True,
            "use_object_obs" : False,
            "use_camera_obs" : True,
            "reward_shaping" : False,
            "camera_depths" : True,
            "camera_heights" : 128,
            "camera_widths" : 128,
            "controller_configs" : suite.load_controller_config(default_controller="JOINT_POSITION")
    }
    ## Check available cpus
    cpu_count = multiprocessing.cpu_count() # Gives 8 on local
    env = DummyVecEnv([lambda : GymWrapperRGBD(suite.make(**env_params), keys=['agentview_image', 'agentview_depth'])])
    # Parse arguments, and load or initialize new PPO instance
    args = parse_arguments()
    # The folders for weights and logs and their filenames
    models_dir = f"./models/PPO-"
    logdir = f"./logs/PPO-"
    #Check if continue training argument is given
    if args.cont != None:
        instance_id = args.cont
        model = stable_baselines3.load(args.model_path, env)
    else:
        instance_id = str(int(time.time()))
        #Create the models directory
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        #Create the logs directory
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        # Initialize policy
        model = stable_baselines3.PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir+instance_id)
    #env = VecNormalize(env) # Is this needed?
    # Train the model
    TIMESTEPS = 10000  #Atomic amount of timetsteps
    try:
        for i in range(TIMESTEPS*100000):
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
            model.save(f"{models_dir+instance_id}/{TIMESTEPS*i}")
    except KeyboardInterrupt:
        env.close()
        print("Closed environment")
    
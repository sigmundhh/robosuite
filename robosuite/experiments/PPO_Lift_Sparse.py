from unicodedata import name
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers import GymWrapperRGBD
import stable_baselines3 as sb3
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

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    Custom callback for plotting rewards from Lift-environment. 
    """
    def __init__(self, verbose=0):
        self.is_tb_set = False
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log additional tensor
        if not self.is_tb_set:
            with self.model.graph.as_default():
                tf.summary.scalar('value_target', tf.reduce_mean(self.model.value_target))
                self.model.summary = tf.summary.merge_all()
            self.is_tb_set = True
        # Log scalar value (here a random variable)
        value = np.random.random()
        summary = tf.Summary(value=[tf.Summary.Value(tag='random_value', simple_value=value)])
        self.locals['writer'].add_summary(summary, self.num_timesteps)
        return True

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
    print(args.cont)
    if args.cont != None:
        instance_id = args.cont
        #Check that model and logs exist for given instance id
        if not os.path.exists(models_dir+instance_id) or not os.path.exists(logdir+instance_id):
            raise ValueError(f"No model or log found for instance id {instance_id}")
        model = sb3.PPO.load(models_dir+instance_id, env)
    else:   # Initialize new model and folders
        instance_id = str(int(time.time()))
        #Check that instance does not already exist (if several computers train in parallel)
        if os.path.exists(models_dir+instance_id) or os.path.exists(logdir+instance_id):
            raise ValueError(f"Model or log already exists for instance id {instance_id}")
        #Create the model and log folders
        os.makedirs(models_dir+instance_id)
        os.makedirs(logdir+instance_id)
        # Initialize policy
        model = sb3.PPO("CnnPolicy", env, verbose=1, tensorboard_log=logdir+instance_id)
    #env = VecNormalize(env) # Is this needed?
    # Train the model
    TIMESTEPS = 10000  #Atomic amount of timesteps
    try:
        for i in range(TIMESTEPS*100000):
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
            model.save(f"{models_dir+instance_id}/{TIMESTEPS*i}")
    except KeyboardInterrupt:
        env.close()
        print("Closed environment")
    
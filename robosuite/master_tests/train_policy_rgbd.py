import robosuite as suite
from robosuite.utils.observables import Observable
from stable_baselines3.ppo import PPO
from robosuite.wrappers import GymWrapperRGBD
import numpy as np
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import SAC

#For visualization
from test_utils import visualize_obs

import wandb
from wandb.keras import WandbCallback

wandb.init(project="Lift_RGBD", entity="sigmundhh")


# Make env
ungym_env = suite.make(
        env_name="Lift",
        robots="IIWA",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_object_obs = False,
        use_camera_obs=True,
        camera_heights=84,                      # image height
        camera_widths=84, 
        reward_shaping= True, 
        camera_depths = True #RGB-D,
    )

#Make a wrapper for the env, specify the modalities we want to use
env = GymWrapperRGBD(
    ungym_env, keys= ['agentview_image', 'agentview_depth']
)


envs = SubprocVecEnv([lambda: env] for i in range(4))

#Init policy
model = SAC("CnnPolicy", env, verbose=1)

#Train policy
model.learn(total_timesteps=200, log_interval=100)

#Save trained policy
model.save("trained_policy_rgbd")

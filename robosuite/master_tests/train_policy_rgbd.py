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

env = GymWrapperRGBD(
    ungym_env, keys= ['agentview_image', 'agentview_depth']
)

# #Init policy
model = SAC("CnnPolicy", env, verbose=1)

# #Train policy
model.learn(total_timesteps=1000, log_interval=1)

# #Save trained policy
model.save("trained_policy_rgbd")

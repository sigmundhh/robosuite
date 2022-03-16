import robosuite as suite
from robosuite.utils.observables import Observable
from stable_baselines3.ppo import PPO
from robosuite.wrappers import GymWrapper
import numpy as np
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import SAC
from test_utils import print_obs_and_action_space

#For visualization
from test_utils import visualize_obs


# Make env
ungym_env = suite.make(
        env_name="Lift",
        robots="IIWA"
    )

env = GymWrapper(
    ungym_env,
    keys=['object-state', 'robot0_proprio-state']
)

print_obs_and_action_space(env)
print(env.modality_dims.keys())

# #Init policy
model = SAC("MlpPolicy", env, verbose=1, tensorboard_log="./sac_flattened_tensorboard/")

# #Train policy
model.learn(total_timesteps=100_000, log_interval=1)

# #Save trained policy
model.save("trained_policy_flattened")

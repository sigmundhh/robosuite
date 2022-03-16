import robosuite as suite
from stable_baselines3.ppo import PPO
from robosuite.wrappers import GymWrapper
import numpy as np
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.vec_env import SubprocVecEnv

# Make Lift env
lift_env = GymWrapper(
    suite.make(
        env_name="Lift",
        robots="IIWA",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        reward_shaping= True
    )
)

# def make_gym_monitor():
#     gym_lift_env = gym_wrapper(lift_env)
#     monitor = Monitor(gym_lift_env)
#     return monitor

# vec_env = SubprocVecEnv([make_gym_monitor])

model = PPO("MlpPolicy", lift_env)

model.learn(total_timesteps = 10000)

obs = lift_env.reset()
for i in range(1000):
    action = model.predict(obs)
    action = action[0]
    #action = [0,0,0,0.1,0,0.1,0.1,0.1]
    print(action)
    obs, reward, ter, info = lift_env.step(action)
    lift_env.render()




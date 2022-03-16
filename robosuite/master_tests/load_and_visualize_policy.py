import gym
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import SAC

model = SAC.load("trained_policy_flattened") #Load from zip

# Make Lift env
ungym_env = suite.make(
        env_name="Lift",
        robots="IIWA",
        has_renderer= True,
        has_offscreen_renderer = False,
        use_camera_obs = False
    )

env = GymWrapper(
    ungym_env,
    keys=['object-state', 'robot0_proprio-state']
)

#env = gym.make("Pendulum-v0") #For test

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
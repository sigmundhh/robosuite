import gym
import numpy as np

from stable_baselines3 import SAC

# Make the env, similar to the gym-wrapped Lift-env
env = gym.make("Pendulum-v0")

# Init of policy
model = SAC("MlpPolicy", env, verbose=1)

#Traning of policy
model.learn(total_timesteps=10000, log_interval=4)
model.save("sac_pendulum") # try to run scrip and see how it is saved, want to do this on Idun

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum") #Is this just a file?

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()
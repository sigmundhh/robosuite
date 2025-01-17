from statistics import mode
import time

from numpy import uint8
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.wrappers.gym_wrapper_rgbd import GymWrapperRGBD
import robosuite.utils.macros as macros
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import os
import sys
import imageio

timesteps_vid = 200
timesteps_trained = 1500000 #Actually 2million
model_id= "PPO-Lift-dense-1651236812" # High reward model
#model_id = "PPO-Lift-dense-1650041358" #Idk
model = PPO.load(f"robosuite/experiments/models/{model_id}/{timesteps_trained}.zip") #Load from zip

macros.IMAGE_CONVENTION = "opencv"

#Specify env according to the loaded model
# RGB-D observations
"""
config = {
    "env_params": {
        "env_name" : "Lift",
        "robots" : "IIWA",
        "has_offscreen_renderer" : True,
        "use_object_obs" : False,
        "use_camera_obs" : True,
        "reward_shaping" : True,
        "camera_depths" : True,
        "camera_heights" : 128,
        "camera_widths" : 128,
        "controller_configs" : suite.load_controller_config(
            default_controller="JOINT_POSITION")
    },
    "observations": ['agentview_image', 'agentview_depth']
}
env = GymWrapperRGBD(suite.make(**config["env_params"]), keys=config["observations"])"""

# Object observations
config = {
    "env_params": {
        "env_name" : "Lift",
        "robots" : "IIWA",
        "has_offscreen_renderer" : True,
        "use_object_obs" : True,
        "use_camera_obs" : False,
        "reward_shaping" : True,
        "controller_configs" : suite.load_controller_config(
            default_controller="OSC_POSE")
    },
}

env = GymWrapper(suite.make(**config["env_params"]))

obs = env.reset()
imgs = []
outputfolder_path = os.path.join(sys.path[0], f'videos/{model_id}/')
outputfolder_path_imgs = os.path.join(sys.path[0], f'images/{model_id}/')
#os.makedirs(outputfolder_path, exist_ok=True)
os.makedirs(outputfolder_path_imgs, exist_ok=True)
#out = cv2.VideoWriter(f'{outputfolder_path}{timesteps_trained}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 25, (128, 128), True)
writer = imageio.get_writer(f'{outputfolder_path}{timesteps_trained}.mp4', fps=20)

print("Writing images to: ", outputfolder_path_imgs)
for t in tqdm(range(timesteps_vid)):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    img = env.env.sim.render(camera_name="frontview", width=1920, height=1080, depth=False, mode="offscreen")
    # Flip the image vertically, as it is currently upside down
    img = cv2.flip(img, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{outputfolder_path_imgs}{t}.png', img)
    #img = env.env.sim._render()
    #out.write(img.astype(uint8))
    #writer.append_data(img)
    if done:
      obs = env.reset()

#out.release()
writer.close()

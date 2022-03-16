
from matplotlib import pyplot as plt
import numpy as np


def visualize_obs(obs):
    """ This, at the moment, will diplay the RGB and D image, 
    but upside down! Fix later at need, works for simple purpose
    of showing content is right. 
    """
    rgb = obs[:,:,:3]
    d = obs[:,:,3]
    plt.subplot(1, 2, 1)
    plt.title('Color image')
    plt.imshow(rgb.astype(np.uint8)) #this gives nothing? Just a white picture?
    plt.subplot(1, 2, 2)
    plt.title('Depth image')
    plt.imshow(d)
    #plt.imshow(obs[:,:,3]) # this gives depth
    plt.show()

def print_obs_and_action_space(env):
    """Input: Gym-env"""
    print("Observation space: ", env.observation_space)
    print("Action space: ",env.action_spec)
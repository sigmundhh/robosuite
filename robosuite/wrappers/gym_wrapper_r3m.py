"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like 
interface.
"""

import numpy as np
from gym import spaces
from gym.core import Env

from robosuite.wrappers import GymWrapper

import torch
import torchvision.transforms as T
from PIL import Image
from r3m import load_r3m


class GymWrapperR3M(GymWrapper, Env):
    """
    Builds upon GymWrapper to provide non-flattened RGB-D observations.

    Does that by only keeping the observations from 'agentview_image' and 'agentview_depth',
    and overriding the flattening originally implemented in GymWrapper.

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env, keys=None)
        width = env.camera_widths[0]    # Assume all cameras have same width
        height = env.camera_heights[0]

        r3m_length = 2048
        proprio_data_length = 40
        high = np.inf * np.ones(r3m_length + proprio_data_length)
        low = -high
        self.observation_space = spaces.Box(low=low, high=high)

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)

        self.r3m = load_r3m("resnet50") # resnet18, resnet34
        self.transforms = T.Compose([T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()]) # ToTensor() divides by 255


    #Untested
    def _make_rgbd_obs(self, obs_dict):
        """
        Returns the RGB-d in the format of a numpy array

        Args:
            obs_dict: not used
        Output:
            np.array: RGBD image in the shape of (height, width, 4)
        """
        
        #rgbd_img = np.concatenate([rgb, d], -1)
        embedding = self._make_r3m_embedding(obs_dict)

        obs = np.concatenate([obs_dict['robot0_proprio-state'], embedding[0]], -1)
        return obs

    def _make_r3m_embedding(self, obs_dict):
        """
        Returns the R3M embedding in the format of a torch tensor
        """
        image = obs_dict["agentview_image"]
        flipped_im = np.flipud(image)
        preprocessed_image = self.transforms(Image.fromarray(flipped_im.astype(np.uint8))).reshape(-1, 3, 224, 224)
        with torch.no_grad():
            embedding = self.r3m(preprocessed_image * 255.0)
        return embedding

    def reset(self):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict.

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        ob_dict = self.env.reset()
        return self._make_rgbd_obs(ob_dict)

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) flattened observations from the environment
                - (float) reward from the environment
                - (bool) whether the current episode is completed or not
                - (dict) misc information
        """
        ob_dict, reward, done, info = self.env.step(action)
        return self._make_rgbd_obs(ob_dict), reward, done, info


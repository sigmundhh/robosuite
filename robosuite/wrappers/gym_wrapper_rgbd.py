"""
This file implements a wrapper for facilitating compatibility with OpenAI gym.
This is useful when using these environments with code that assumes a gym-like 
interface.
"""

import numpy as np
from gym import spaces
from gym.core import Env

from robosuite.wrappers import GymWrapper


class GymWrapperRGBD(GymWrapper, Env):
    """
    Initializes the Gym wrapper. Mimics many of the required functionalities of the Wrapper class
    found in the gym.core module

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
        width = 84
        height = 84
        n_channels = 4 #RGB-D
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, n_channels), dtype=np.uint8)

        low, high = self.env.action_spec
        self.action_space = spaces.Box(low=low, high=high)


    #Untested
    def _make_rgbd_obs(self, obs_dict):
        """
        Returns the RGB-d in the format of a numpy array

        Args:
            obs_dict: not used
        Output:
            np.array: RGBD image in the shape of (84, 84, 4)
        """
        ## Rewrite: add without loop, make sure they are added in correct order
        rgb = obs_dict["agentview_image"] # Assume in rgb format
        d = obs_dict["agentview_depth"]
        rgbd_img = np.concatenate([rgb, d], -1)
        #print("img.shape after _make_rgbd_obs: ", rgbd_img.shape)
        return rgbd_img



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


# Image Preprocessing

# Importing the libraries
import numpy as np
import cv2
from gym.core import ObservationWrapper
from gym.spaces.box import Box


# Preprocessing the Images

class PreprocessImage(ObservationWrapper):
    def __init__(self, env, height=64, width=64, grayscale=True, crop=lambda img: img):
        super().__init__(env)
        self.img_size = (height, width)
        self.grayscale = grayscale
        self.crop = crop
        n_colors = 1 if self.grayscale else 3
        self.observation_space = Box(0.0, 1.0, [n_colors, height, width])

    def observation(self, img):
        img = self.crop(img)
        img = cv2.resize(src=img, dsize=self.img_size)
        if self.grayscale:
            img = img.mean(-1, keepdims=True)
        img = np.transpose(img, (2, 0, 1))
        img = img.astype('float32') / 255.
        return img

    def reset(self):
        """
        We need to override the reset, because it should return a tuple of
        (obs, info). However, vizdoomgym only return obs.

        See:
        https://github.com/shakenes/vizdoomgym/blob/
        3ab9052de6fb4172544d35103857e3577526b57c/vizdoomgym/envs/vizdoomenv.py#L129
        """
        obs = self.env.reset()
        # Return observation and info=None
        return self.observation(obs), None
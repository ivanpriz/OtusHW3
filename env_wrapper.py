import cv2
import gymnasium as gym
import numpy as np
from collections import deque
from image_utils import to_grayscale, zero_center, crop


class EnvironmentWrapper(gym.Wrapper):
    def __init__(self, env, stack_size):
        super().__init__(env)
        self.stack_size = stack_size
        self.frames = deque([], maxlen=stack_size)

    def reset(self, *args):
        state, info = self.env.reset()

        # first 50 steps have bad camera
        for i in range(50):
            s, r, terminated, truncated, info = self.env.step([0.0, 0.0, 1.0])

        for _ in range(self.stack_size):
            self.frames.append(self.preprocess(state))
        return self.state(), info

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        # self.env.env.viewer.window.dispatch_events()
        preprocessed_state = self.preprocess(state)

        self.frames.append(preprocessed_state)
        return self.state(), reward, terminated, truncated, info

    def state(self):
        return np.stack(self.frames, axis=0)

    def preprocess(self, state):
        state = state[:84, 6:90]  # CarRacing-v2-specific cropping
        # img = cv2.resize(img, dsize=(84, 84)) # or you can simply use rescaling

        img = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY) / 255.0
        # preprocessed_state = to_grayscale(state)
        preprocessed_state = zero_center(img)
        # preprocessed_state = crop(preprocessed_state)
        return preprocessed_state

    def get_state_shape(self):
        return self.stack_size, 84, 84

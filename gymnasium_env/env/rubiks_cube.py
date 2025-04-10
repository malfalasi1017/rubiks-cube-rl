from enum import Enum
from gymnasium import spaces

import open3d as o3d

import gymnasium as gym
import numpy as np


class Actions(Enum):
    FRONT_CW = 0
    FRONT_CCW = 1
    RIGHT_CW = 2
    RIGHT_CCW = 3
    UP_CW = 4
    UP_CCW = 5
    LEFT_CW = 6
    LEFT_CCW = 7
    DOWN_CW = 8
    DOWN_CCW = 9
    BACK_CW = 10
    BACK_CCW = 11

class RubiksCubeEnv(gym.Env):
    '''
    Rubik's Cube environment for reinforcement learning.
    '''
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30
    }

    COLOR_MAP = {
        0: (255, 255, 255),  # White
        1: (255, 165, 0),    # Orange
        2: (0, 0, 255),      # Blue
        3: (255, 0, 0),      # Red
        4: (0, 255, 0),      # Green
        5: (255, 255, 0)     # Yellow
    }

    def __init__(self, scrambles=5, render_mode=None):
        super().__init__()

        # Number of scrambles to perform
        self.scrambles = scrambles

        # Action Space: 12 possible moves (6 faces * 2 directions)
        self.action_space = spaces.Discrete(12)

        # Observation Space: 54 squares (6 faces * 9 squares)
        self.observation_space = spaces.Box(low=0, high=5, shape=(54,), dtype=np.uint8)

        # Rendering attributes
        self.render_mode = render_mode
        self.window = None
        self.clock = None

        # Cube state attributes
        self.cube = None
        self.steps = 0
        self.solved_state = None

        # Add these lines:
        self.vis = None
        self.vis_initialized = False

    def reset(self, seed=None, options=None):
        pass

    def step(self, action):
        pass


    # Rendering the environment
    def render(self):
        pass



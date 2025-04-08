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

    def reset():
        pass

    def step(self, action):
        pass

    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    def _is_solved(self):
        pass

    def _count_correct_squares(self):
        pass

    def _apply_action(self, action):
        pass

    # Rendering the environment
    def render(self):
        if self.render_mode is None:
            return
        
        if self.cube is None:
            return
        
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Rubik's Cube", width=800, height=600)

        cube_meshes = self._create_cube_meshes()
        for mesh in cube_meshes:
            vis.add_geometry(mesh)

        # set default camera
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0.5, 0.5, -0.5])
        ctr.set_up([0, 1, 0])
        ctr.set_lookat([0, 0, 0])

        # Run the visualizer
        vis.run()
        vis.destroy_window()

    def _create_cube_meshes(self):
        meshes = []

        size = 0.95

        positions = []
        for x in range(-1, 2):
            for y in range(-1, 2):
                for z in range(-1, 2):
                    positions.append([x, y, z])

        face_colors = [[c[0]/255, c[1]/255, c[2]/255] for c in face_colors]

        for i, pos in enumerate(positions):
            mesh = o3d.geometry.TriangleMesh.create_box(width=size, height=size, depth=size)
            mesh.translate([pos[0], pos[1], pos[2]])

            vertex_colors = []
            for vertex in range(len(mesh.vertices)):
                vertex_colors.append([0.2, 0.2, 0.2])

            if self.cube is not None:
                pass

            mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
            mesh.compute_vertex_normals()
            meshes.append(mesh)

        return meshes




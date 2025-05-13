from enum import Enum
from gymnasium import spaces
from typing import Optional

import gymnasium as gym
import numpy as np

import pygame

COLOR_MAP = {
        0: (255, 255, 255),  # White
        1: (255, 165, 0),    # Orange
        2: (0, 0, 255),      # Blue
        3: (255, 0, 0),      # Red
        4: (0, 255, 0),      # Green
        5: (255, 255, 0)     # Yellow
    }

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

    def __init__(self, scrambles=5, max_steps=100):
        super().__init__()

        # Number of scrambles to perform
        self.scrambles = scrambles

        # Maximum number of steps per episode
        self.max_steps = max_steps

        # Action Space: 12 possible moves (6 faces * 2 directions)
        self.action_space = spaces.Discrete(12)

        # Observation Space: 54 squares (6 faces * 9 squares)
        self.observation_space = spaces.Box(low=0, high=5, shape=(54,), dtype=np.uint8)

        self._reset_cube()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self._reset_cube()

        # Perform scrambles
        for _ in range(self.scrambles):
            action = self.action_space.sample()
            self._apply_action(action)

        # Reset step counter
        self.current_step = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Apply the action
        self._apply_action(action)

        # Increment step counter
        self.current_step += 1

        # Check if the cube is solved
        solved = self._is_solved()

        if solved:
            reward = 1.0
            terminated = True
        elif self.current_step >= self.max_steps:
            reward = -1.0  # Penalty for exceeding max steps
            terminated = True
        else:
            reward = -0.01  # Small penalty for each step
            terminated = False

        truncated = False

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    # Rendering the environment
    def render(self):
        SQUARE_SIZE = 50
        FACE_SIZE = SQUARE_SIZE * 3
        MARGIN = 10
        GRID_WIDTH = 3
        GRID_HEIGHT = 2

        if not hasattr(self, 'pygame_initialized'):
            pygame.init()
            self.pygame_initialized = True
            self.screen = None

        win_width = GRID_WIDTH * FACE_SIZE + (GRID_WIDTH + 1) * MARGIN
        win_height = GRID_HEIGHT * FACE_SIZE + (GRID_HEIGHT + 1) * MARGIN

        if self.screen is None or self.screen.get_width() != win_width or self.screen.get_height() != win_height:
            self.screen = pygame.display.set_mode((win_width, win_height))
            pygame.display.set_caption("Rubik's Cube")

        self.screen.fill((0, 0, 0))

        face_positions = {
            4: (0, 0), # Up face - Top Left
            2: (0, 1), # Back face - Top Right
            3: (0, 2), # Left face - middle Left
            0: (1, 0), # Front face - Middle Center
            5: (1, 1), # Down face - Middle Right
            1: (1, 2), # Right face - Bottom Right
        }

        for face_idx, (row, col) in face_positions.items():
            face_x = MARGIN + col * (FACE_SIZE + MARGIN)
            face_y = MARGIN + row * (FACE_SIZE + MARGIN)

            # Face labels
            font = pygame.font.SysFont(None, 24)
            face_names = ["Front", "Right", "Back", "Left", "Up", "Down"]
            text_surface = font.render(face_names[face_idx], True, (255, 255, 255))
            self.screen.blit(text_surface, (face_x, face_y - 25))

            for i in range(3):
                for j in range(3):
                    square_idx  = i * 3 + j
                    color_val = self.cube_state[face_idx][square_idx]
                    color = COLOR_MAP[color_val]

                    square_x = face_x + j * SQUARE_SIZE
                    square_y = face_y + i * SQUARE_SIZE

                    # Draw colored square with border
                    pygame.draw.rect(self.screen, color, (square_x, square_y, SQUARE_SIZE, SQUARE_SIZE))
                    pygame.draw.rect(self.screen, (50, 50, 50), (square_x, square_y, SQUARE_SIZE, SQUARE_SIZE), 1)

            
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self.pygame_initialized = False

        pygame.time.wait(50)
        return self.screen 

    def _reset_cube(self):
        self.cube_state = np.array([
            [0] * 9,  # Front Face - White
            [1] * 9,  # Right Face - Orange
            [2] * 9,  # Back Face - Blue
            [3] * 9,  # Left Face - Red
            [4] * 9,  # Up Face - Green
            [5] * 9,  # Down Face - Yellow
        ])
        

    def _get_obs(self):
        return self.cube_state.flatten()

    def _get_info(self):
        solved_faces = 0
        for face in range(6):
            if self._is_face_solved(face):
                solved_faces += 1
        
        return {
            "solved_faces": solved_faces,
            "is_solved": self._is_solved(),
        }
    
    def _is_solved(self):
        for face in range(6):
            if not self._is_face_solved(face):
                return False
        return True
    
    def _is_face_solved(self, face):
        first_color = self.cube_state[face][0]
        return all(color == first_color for color in self.cube_state[face])


    def _apply_action(self, action):
        face = action // 2

        is_clockwise = action % 2 == 0

        if is_clockwise:
            self._rotate_face_cw(face)
        else:
            for _ in range(3):
                self._rotate_face_cw(face)

    def _rotate_face_cw(self, face):

        cube_copy = self.cube_state.copy()

        # Rotate the corners of the face
        self.cube_state[face][0] = cube_copy[face][6]
        self.cube_state[face][2] = cube_copy[face][0]
        self.cube_state[face][8] = cube_copy[face][2]
        self.cube_state[face][6] = cube_copy[face][8]

        # Rotate the edges of the face
        self.cube_state[face][1] = cube_copy[face][3]
        self.cube_state[face][5] = cube_copy[face][1]
        self.cube_state[face][7] = cube_copy[face][5]
        self.cube_state[face][3] = cube_copy[face][7]

        if face == 0:  # Front face
            for i in range(3):
                self.cube_state[1][0 + i * 3] = cube_copy[4][6 + i]  # Up -> Right
                self.cube_state[5][0 + i] = cube_copy[1][0 + i * 3]  # Right -> Down
                self.cube_state[3][2 + i * 3] = cube_copy[5][0 + i]  # Down -> Left
                self.cube_state[4][6 + i] = cube_copy[3][2 + i * 3]  # Left -> Up
            
        elif face == 1:  # Right face
            for i in range(3):
                self.cube_state[2][0 + i * 3] = cube_copy[4][2 + i * 3]  # Up -> Back
                self.cube_state[5][2 + i * 3] = cube_copy[2][0 + i * 3]  # Back -> Down
                self.cube_state[0][2 + i * 3] = cube_copy[5][2 + i * 3]  # Down -> Front
                self.cube_state[4][2 + i * 3] = cube_copy[0][2 + i * 3]  # Front -> Up
        
        elif face == 2:  # Back face
            for i in range(3):
                self.cube_state[3][0 + i * 3] = cube_copy[4][2 - i]  # Up -> Left
                self.cube_state[5][8 - i] = cube_copy[3][0 + i * 3]  # Left -> Down
                self.cube_state[1][2 + i * 3] = cube_copy[5][8 - i]  # Down -> Right
                self.cube_state[4][2 - i] = cube_copy[1][2 + i * 3]  # Right -> Up
        
        elif face == 3:  # Left face
            for i in range(3):
                self.cube_state[0][0 + i * 3] = cube_copy[4][0 + i * 3]  # Up -> Front
                self.cube_state[5][0 + i * 3] = cube_copy[0][0 + i * 3]  # Front -> Down
                self.cube_state[2][2 + i * 3] = cube_copy[5][0 + i * 3]  # Down -> Back
                self.cube_state[4][0 + i * 3] = cube_copy[2][2 + i * 3]  # Back -> Up
        
        elif face == 4:  # Up face
            for i in range(3):
                self.cube_state[1][i] = cube_copy[2][i]  # Back -> Right
                self.cube_state[0][i] = cube_copy[1][i]  # Right -> Front
                self.cube_state[3][i] = cube_copy[0][i]  # Front -> Left
                self.cube_state[2][i] = cube_copy[3][i]  # Left -> Back
        
        elif face == 5:  # Down face
            for i in range(3):
                self.cube_state[1][6 + i] = cube_copy[0][6 + i]  # Front -> Right
                self.cube_state[2][6 + i] = cube_copy[1][6 + i]  # Right -> Back
                self.cube_state[3][6 + i] = cube_copy[2][6 + i]  # Back -> Left
                self.cube_state[0][6 + i] = cube_copy[3][6 + i]  # Left -> Front

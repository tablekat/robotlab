import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RobotEnv(gym.Env):
    """2D Robot simulation environment."""
    
    def __init__(self, width=800, height=600):
        super().__init__()
        self.width = width
        self.height = height
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        
        # Robot properties
        self.robot_size = 20
        self.robot_pos = np.array([width/2, height/2])
        self.robot_angle = 0
        self.robot_speed = 5
        
        # Environment objects (obstacles, targets, etc.)
        self.objects = []
        self.targets = []
        
        # Action space: [move_forward, turn_left, turn_right, interact]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )
        
        # Observation space: robot state + environment state
        self.observation_space = spaces.Dict({
            'robot_state': spaces.Box(
                low=np.array([0, 0, -np.pi]),
                high=np.array([width, height, np.pi]),
                dtype=np.float32
            ),
            'vision': spaces.Box(
                low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
            ),
            'task_embedding': spaces.Box(
                low=-np.inf, high=np.inf, shape=(128,), dtype=np.float32
            )
        })
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.robot_pos = np.array([self.width/2, self.height/2])
        self.robot_angle = 0
        self.objects = self._generate_random_objects()
        self.targets = self._generate_random_targets()
        return self._get_obs(), {}
    
    def step(self, action):
        # Process action
        move_forward, turn_left, turn_right, interact = action
        
        # Update robot position and orientation
        self.robot_angle += (turn_right - turn_left) * 0.1
        movement = move_forward * self.robot_speed
        self.robot_pos += np.array([
            np.cos(self.robot_angle) * movement,
            np.sin(self.robot_angle) * movement
        ])
        
        # Keep robot within bounds
        self.robot_pos = np.clip(
            self.robot_pos,
            [self.robot_size, self.robot_size],
            [self.width - self.robot_size, self.height - self.robot_size]
        )
        
        # Check for collisions and interactions
        reward = self._calculate_reward(interact)
        done = self._check_done()
        
        # Render if display is enabled
        self._render_frame()
        
        return self._get_obs(), reward, done, False, {}
    
    def _get_obs(self):
        return {
            'robot_state': np.array([
                self.robot_pos[0],
                self.robot_pos[1],
                self.robot_angle
            ], dtype=np.float32),
            'vision': self._get_vision(),
            'task_embedding': np.zeros(128)  # Placeholder for task embedding
        }
    
    def _render_frame(self):
        self.screen.fill((255, 255, 255))
        
        # Draw robot
        pygame.draw.circle(
            self.screen,
            (0, 0, 255),
            self.robot_pos.astype(int),
            self.robot_size
        )
        # Draw direction indicator
        end_pos = self.robot_pos + np.array([
            np.cos(self.robot_angle) * self.robot_size,
            np.sin(self.robot_angle) * self.robot_size
        ])
        pygame.draw.line(
            self.screen,
            (255, 0, 0),
            self.robot_pos.astype(int),
            end_pos.astype(int),
            2
        )
        
        # Draw objects and targets
        for obj in self.objects:
            pygame.draw.rect(self.screen, (100, 100, 100), obj)
        for target in self.targets:
            pygame.draw.circle(self.screen, (0, 255, 0), target[:2], target[2])
        
        pygame.display.flip()
        self.clock.tick(60)
    
    def _get_vision(self):
        # Capture the robot's view and return as numpy array
        view_surface = pygame.Surface((84, 84))
        view_surface.blit(
            self.screen,
            (0, 0),
            (
                self.robot_pos[0] - 42,
                self.robot_pos[1] - 42,
                84,
                84
            )
        )
        return pygame.surfarray.array3d(view_surface)
    
    def _generate_random_objects(self):
        # Generate random obstacles
        objects = []
        for _ in range(5):
            pos = np.random.randint(0, [self.width, self.height])
            size = np.random.randint(20, 50, size=2)
            objects.append(pygame.Rect(pos[0], pos[1], size[0], size[1]))
        return objects
    
    def _generate_random_targets(self):
        # Generate random target points
        targets = []
        for _ in range(3):
            pos = np.random.randint(0, [self.width, self.height])
            radius = np.random.randint(10, 30)
            targets.append([pos[0], pos[1], radius])
        return targets
    
    def _calculate_reward(self, interact):
        # Basic reward function
        reward = -0.01  # Small time penalty
        
        # Check for target reaching
        for target in self.targets:
            dist = np.linalg.norm(self.robot_pos - target[:2])
            if dist < target[2] and interact > 0.5:
                reward += 1.0
        
        # Penalty for collision with obstacles
        for obj in self.objects:
            if obj.collidepoint(self.robot_pos):
                reward -= 0.5
        
        return reward
    
    def _check_done(self):
        # Episode ends if all targets are reached or max steps exceeded
        return False  # Simplified for now
    
    def close(self):
        pygame.quit() 
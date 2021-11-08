import numpy as np
import random

def generate_random_obstacle(env_min, env_max, max_rad):
    x_pos = random.uniform(env_min + max_rad, env_max - max_rad)
    y_pos = random.uniform(env_min + max_rad, env_max - max_rad)
    rad = random.uniform(0.05, max_rad)
    area = np.pi * (rad ** 2)
    return ([x_pos, y_pos, rad], area)

class EnvConfig:
    def __init__(self, clutter_ratio = 0.1):
        self.env_range = [0, 30]
        self.no_obstacles = 0
        self.max_obs_radius = 0.5
        self.obs = []
        area = 15 ** 2
        obs_area = 0
        while obs_area < (area * clutter_ratio):
            temp_ob, ob_area = generate_random_obstacle(self.env_range[0] + 7.5, self.env_range[1] - 7.5, self.max_obs_radius)
            self.obs.append(temp_ob)
            obs_area += ob_area
        self.obs = np.array(self.obs)
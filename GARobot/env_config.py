import numpy as np
import random

def generate_random_obstacle(env_min, env_max, max_rad):
    x_pos = random.uniform(env_min + max_rad, env_max - max_rad)
    y_pos = random.uniform(env_min + max_rad, env_max - max_rad)
    rad = round(random.uniform(0.1, max_rad), 2)
    area = np.pi * (rad ** 2)
    return ([x_pos, y_pos, rad], area)

class EnvConfig:
    def __init__(self, clutter_pct=10):
        self.env_range = [0, 10]
        self.no_obstacles = 0
        self.max_obs_radius = 0.25
        self.obs = []
        area = (9) ** 2
        obs_area = 0
        while obs_area < (area * (clutter_pct/100)):
            temp_ob, ob_area = generate_random_obstacle(self.env_range[0] + 0.5, self.env_range[1] - 0.5, self.max_obs_radius)
            self.obs.append(temp_ob)
            obs_area += ob_area
        self.obs = np.array(self.obs)
import numpy as np
import random

def generate_random_obstacle(env_range_x, env_range_y, min_rad, max_rad):
    x_pos = random.uniform(env_range_x[0] + max_rad, env_range_x[1] - max_rad)
    y_pos = random.uniform(env_range_y[0] + max_rad, env_range_y[1] - max_rad)
    rad = round(random.uniform(min_rad, max_rad), 2)
    area = np.pi * (rad ** 2)
    return ([x_pos, y_pos, rad], area)

class EnvConfig:
    def __init__(self, clutter_pct=15):
        self.env_range = [0, 10]
        self.env_range_x = [0.5, 9.5]
        self.env_range_y = [2.5, 7.5]
        self.no_obstacles = 0
        self.min_obs_radius = 0.15
        self.max_obs_radius = 0.3
        self.obs = []
        area = (self.env_range_x[1] - self.env_range_x[0]) * (self.env_range_y[1] - self.env_range_y[0])
        obs_area = 0
        while obs_area < (area * (clutter_pct/100)):
            temp_ob, ob_area = generate_random_obstacle(self.env_range_x, self.env_range_y, self.min_obs_radius, self.max_obs_radius)
            self.obs.append(temp_ob)
            obs_area += ob_area
        self.obs = np.array(self.obs)
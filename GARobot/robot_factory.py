import numpy as np
import random

class BaseRobotConfig:
    def __init__(self):
        # Common parameters for all robots
        self.robot_radius = 0.05 # For collision check
        self.max_speed = 1 # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_accel = 0.5  # [m/ss]
        self.v_resolution = 0.01  # [m/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 0.1 # [s]

class RobotGenome():
    
    def __init__(self):
        # Parameters to tune for GARobot
        # Also used to check if goal is reached in both types
        self.to_goal_cost_gain = 2
        self.obstacle_cost_gain = 2
        self.obstacle_sphere_of_influence = 0.8 # [m] for obstacle potential field
    
    @staticmethod
    def create_random_genome():
        genome = RobotGenome()
        genome.to_goal_cost_gain = random.randrange(0, 10)
        genome.obstacle_cost_gain = random.randrange(0, 10)
        genome.obstacle_sphere_of_influence = random.uniform(0.7, 2)
        return genome

class Robot(BaseRobotConfig):

    def __init__(self, genome, init_state):
        self.genome = genome
        self.init_state = init_state
        self.state = init_state
        super(Robot, self).__init__()

        self.reset_robot()

    def reset_robot(self):
        self.trajectory = np.array(self.init_state)
        self.trajectory_cost = 0
        self.reached_goal = False

        self.time_steps = 0
        self.no_collisions = 0
        self.distance_travelled = 0
    
    @staticmethod
    def create_robot(sx=0.0, sy=15.0):
        # initial state [x(m), y(m), v_x(m/s), v_y(m/s)]
        state = np.array([sx, sy, 0, 0.0])
        genome = RobotGenome.create_random_genome()
        return Robot(genome, state)
    
    def update_fitness_params(self, u, no_collisions):
        self.distance_travelled += (np.sqrt(u[0]**2 + u[1]**2) / self.dt)
        self.no_collisions += no_collisions
        self.time_steps += 1
import numpy as np
import random

def crossover_genome_values(g_val_1, g_val_2, default_val=0.5):
    randn1p1 = 1 if random.random() <= 0.5 else -1
    crossover_val = ((g_val_1 + g_val_2)/2) + (abs(g_val_1 - g_val_2) * randn1p1)

    if(crossover_val < 0):
        crossover_val = default_val

    return crossover_val

class BaseRobotConfig:
    def __init__(self):
        # Common parameters for all robots
        self.robot_radius = 0.1 # For collision check
        self.max_speed = 0.5 # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.v_resolution = 0.075  # [m/s]
        self.dt = 0.5  # [s] Time tick for motion prediction
        self.predict_time = 1 # [s]

class RobotGenome:
    
    def __init__(self):
        # Parameters to tune for GARobot
        # Also used to check if goal is reached in both types
        self.to_goal_cost_gain = 2
        self.obstacle_cost_gain = 2
        self.obstacle_sphere_of_influence = 0.2 # [m] for obstacle potential field

    def mutate(self, delta):
        randn1p1 = 1 if random.random() < 0.5 else -1
        self.to_goal_cost_gain += self.to_goal_cost_gain * delta * randn1p1
        
        randn1p1 = 1 if random.random() < 0.5 else -1
        self.obstacle_cost_gain += self.obstacle_cost_gain * delta * randn1p1
        
        randn1p1 = 1 if random.random() < 0.5 else -1
        self.obstacle_sphere_of_influence += self.obstacle_sphere_of_influence * delta * randn1p1
    
    @staticmethod
    def create_random_genome():
        genome = RobotGenome()
        genome.to_goal_cost_gain = random.uniform(1, 10)
        genome.obstacle_cost_gain = random.uniform(1, 10)
        genome.obstacle_sphere_of_influence = random.uniform(0, 3)
        return genome
    
    @staticmethod
    def crossover_genomes(genome_1, genome_2):
        new_genome = RobotGenome()

        new_genome.to_goal_cost_gain = crossover_genome_values(genome_1.to_goal_cost_gain, genome_2.to_goal_cost_gain)
        new_genome.obstacle_cost_gain = crossover_genome_values(genome_1.obstacle_cost_gain, genome_2.obstacle_cost_gain)
        new_genome.obstacle_sphere_of_influence = crossover_genome_values(genome_1.obstacle_sphere_of_influence, genome_2.obstacle_sphere_of_influence)

        return new_genome

class Robot(BaseRobotConfig):

    def __init__(self, genome, init_state):
        self.genome = genome
        self.init_state = init_state
        self.state = init_state.copy()
        super(Robot, self).__init__()

        self.reset_robot()

        self.fitness = 0

    def reset_robot(self):
        self.reset_robot_state()
        self.reset_fitness_params()

    def reset_robot_state(self):
        self.state = self.init_state.copy()
        self.trajectory = np.array(self.init_state)
        self.trajectory_cost = 0
        self.reached_goal = False

    def reset_fitness_params(self):
        self.time_steps = 0
        self.no_collisions = 0
        self.distance_travelled = 0
        self.distance_from_goal = 0
    
    @staticmethod
    def create_robot(start_pos):
        # initial state [x(m), y(m), v_x(m/s), v_y(m/s)]
        state = np.array([start_pos[0], start_pos[1], 0, 0.0])
        genome = RobotGenome.create_random_genome()
        return Robot(genome, state)
    
    def update_fitness_params(self, u, no_collisions):
        self.distance_travelled += (np.sqrt(u[0]**2 + u[1]**2) / self.dt)
        self.no_collisions += no_collisions
        self.time_steps += 1
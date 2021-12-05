'''
Code to create and modify the robot, including the individual genetic operators.

Author: Megnath Ramesh
'''

import numpy as np
import random

# Method to crossover two genome values of robots
def crossover_genome_values(g_val_1, g_val_2, min_val=0.1):
    randn1p1 = 1 if random.random() <= 0.5 else -1
    crossover_val = ((g_val_1 + g_val_2)/2) + (abs(g_val_1 - g_val_2) * randn1p1)

    if(crossover_val < min_val):
        crossover_val = min_val

    return crossover_val

# Configuration of the base capabilities of the robot
# Remains constant during the experiments
class BaseRobotConfig:
    def __init__(self):
        # Common parameters for all robots
        self.robot_radius = 0.1 # For collision check
        self.max_speed = 0.45 # [m/s]
        self.min_speed = -0.45  # [m/s]
        self.v_resolution = 0.075  # [m/s]
        self.dt = 0.5  # [s] Time tick for motion prediction
        self.predict_time = 1.5 # [s]

# Class to define the robot's genome
class RobotGenome:
    def __init__(self):
        # Parameters to tune for GARobot
        self.to_goal_cost_gain = 2
        self.obstacle_cost_gain = 2
        self.obstacle_sphere_of_influence = 0.2

    # Method to mutate the genome
    def mutate(self, delta):
        randn1p1 = 1 if random.random() < 0.5 else -1
        self.to_goal_cost_gain += self.to_goal_cost_gain * delta * randn1p1
        
        randn1p1 = 1 if random.random() < 0.5 else -1
        self.obstacle_cost_gain += self.obstacle_cost_gain * delta * randn1p1
        
        randn1p1 = 1 if random.random() < 0.5 else -1
        self.obstacle_sphere_of_influence += self.obstacle_sphere_of_influence * delta * randn1p1
    
    # Method to create a random genome during robot construction
    @staticmethod
    def create_random_genome():
        genome = RobotGenome()
        genome.to_goal_cost_gain = random.uniform(0.1, 1)
        genome.obstacle_cost_gain = random.uniform(0.1, 1)
        genome.obstacle_sphere_of_influence = random.uniform(0.1, 1)
        return genome
    
    # Method to crossover two genomes to create a new genome
    @staticmethod
    def crossover_genomes(genome_1, genome_2):
        new_genome = RobotGenome()

        new_genome.to_goal_cost_gain = crossover_genome_values(genome_1.to_goal_cost_gain, genome_2.to_goal_cost_gain)
        new_genome.obstacle_cost_gain = crossover_genome_values(genome_1.obstacle_cost_gain, genome_2.obstacle_cost_gain)
        new_genome.obstacle_sphere_of_influence = crossover_genome_values(genome_1.obstacle_sphere_of_influence, genome_2.obstacle_sphere_of_influence)

        return new_genome

# Class to represent the robot, its genome and genetic fitness in the different experiments.
class Robot(BaseRobotConfig):

    def __init__(self, genome, init_state):
        self.genome = genome
        self.init_state = init_state
        self.state = init_state.copy()
        super(Robot, self).__init__()

        self.reset_robot()

        self.fitness = 0
        self.objective_val = 0

    # Clear all information of the robot
    def reset_robot(self):
        self.reset_robot_state()
        self.reset_fitness_params()

    # Reset the state information of the robot while maintaining 
    def reset_robot_state(self):
        self.state = self.init_state.copy()
        self.trajectory = np.array(self.init_state)
        self.trajectory_cost = 0
        self.reached_goal = False

    # Reset the metrics that determine the fitness of the robot
    def reset_fitness_params(self):
        self.time_steps = 0
        self.no_collisions = 0
        self.distance_travelled = 0
        self.distance_from_goal = 0
    
    # Method to create a robot with a random genome configuration
    @staticmethod
    def create_robot(start_pos):
        # initial state [x(m), y(m), v_x(m/s), v_y(m/s)]
        state = np.array([start_pos[0], start_pos[1], 0.0, 0.0])
        genome = RobotGenome.create_random_genome()
        return Robot(genome, state)
    
    # Update fitness metrics of the robot
    def update_fitness_metrics(self, u, no_collisions):
        self.distance_travelled += (np.sqrt(u[0]**2 + u[1]**2) * self.dt)
        self.no_collisions += no_collisions
        self.time_steps += 1
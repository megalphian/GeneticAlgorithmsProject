import numpy as np
from numpy.lib.function_base import select
import random

from controller import run_generation
from env_config import EnvConfig
from robot_factory import Robot

goal_dist_gain = 100

class RobotType:
    
    # Robot profiles and corresponding objective gains
    # [Collision, Time, Distance]
    SAFE = [100, 1, 1]
    DIRECT = [10, 10, 1]
    FAST = [10, 1, 10]

class GARobotConfig:
    def __init__(self):
        self.num_gens = 1
        self.runs_per_gen = 3
        self.robot_type_gains = RobotType.SAFE

def evaluate(robots, robot_type_gains):
    robot_obj_lookup = dict()
    total_obj_val = 0
    highest_obj_val = 0

    for robot in robots:
        raw_objective = robot.no_collisions * robot_type_gains[0] + \
                              robot.time_steps * robot_type_gains[1] + \
                              robot.distance_travelled * robot_type_gains[2] + \
                              robot.distance_from_goal * goal_dist_gain
        robot_obj_lookup[robot] = raw_objective
        total_obj_val += raw_objective
        if(highest_obj_val < raw_objective):
            highest_obj_val = raw_objective
    
    for robot,obj in robot_obj_lookup.items():
        robot.fitness = highest_obj_val - obj + 1 

    return total_obj_val

def reproduce(robots, no_reproduce):
    # # Idk how to use this from the paper. Just gonna pick the best pop
    # selection_prob = 0.6 

    fitness_list = []
    selected = []

    for robot in robots:
        fitness_list.append(robot.fitness)
    
    robots_copy = robots.copy()
    
    for i in range(no_reproduce):
        index = fitness_list.index(max(fitness_list))
        
        fitness_list.pop(index)
        robot = robots_copy.pop(index)
        
        selected.append(robot)
    
    return selected

def crossover(robots):
    crossover_prob = 0.6
    pass

def mutate(robots):
    mutation_prob = 0.5
    pass

def garobot(pop_size, goal, config):
    
    # Build environment
    env_config = EnvConfig()

    # Build initial population
    robots = []
    gen_obj_vals = []

    for i in range(pop_size):
        robots.append(Robot.create_robot(0, 5))

    for i in range(config.num_gens):
        print('Generation',i+1)
        for j in range(config.runs_per_gen):
            print('Run',j+1)
            # Run the motions for the generation
            run_generation(robots, goal, env_config, show_animation=True)

            # Recreate the environment with new obstacles
            env_config = EnvConfig()

        # Evaluate the objective value of the population and record
        total_obj_val = evaluate(robots, config.robot_type_gains)
        gen_obj_vals.append(total_obj_val)

        # Apply the GA parameters to the robots
        selected = reproduce(robots, int(pop_size/2))
        crossover(robots)
        mutate(robots)

        for robot in robots:
            robot.reset_fitness_params()
    
    print('GARobot done!')
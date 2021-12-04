'''
Code to run the genetic algorithm approach GAROBOT to evolve robots 
capable of traversing cluttered environments.

Author: Megnath Ramesh
'''

import math
import random

from controller import run_generation
from env_config import EnvConfig
from robot_factory import Robot, RobotGenome
from get_metrics import GARobotMetrics

# Defines the type of robots to evolve in GAROBOT
class RobotType:
    
    # Robot profiles and corresponding objective gains
    # [Collision, Time, Distance, Goal]
    SAFE = [100, 1, 1, 10]
    FAST = [10, 100, 1, 10]
    DIRECT = [10, 1, 10, 10]
    REACH = [10, 0, 0, 100]

# Configure the parameters of the genetic algorithm
class GARobotConfig:
    
    def __init__(self, num_gens=10, \
                runs_per_gen=3, robot_type_gains=RobotType.SAFE, \
                fixed=True, clutter_pct=10):
        self.num_gens = num_gens
        self.runs_per_gen = runs_per_gen
        self.robot_type_gains = robot_type_gains
        self.fixed = fixed

        self.clutter_pct = clutter_pct

# Method to evaluate the population of robots by computing the objective function
def evaluate(robots, robot_type_gains):
    robot_obj_lookup = dict()
    total_obj_val = 0
    highest_obj_val = 0

    # Calculate the objective value for all robots and get the highest objective value
    for robot in robots:
        raw_objective = robot.no_collisions * robot_type_gains[0] + \
                            robot.time_steps * robot_type_gains[1] + \
                            robot.distance_travelled * robot_type_gains[2] + \
                            robot.distance_from_goal * robot_type_gains[3]
        robot_obj_lookup[robot] = raw_objective
        total_obj_val += raw_objective
        
        # Record the highest objective value to compute fitness
        if(highest_obj_val < raw_objective):
            highest_obj_val = raw_objective
    
    # Compute the fitness of each robot using the highest objective value
    for robot,obj in robot_obj_lookup.items():
        robot.objective_val = obj
        robot.fitness = highest_obj_val - obj + 1 

    return total_obj_val

# Get the number of robots to select in the selection procedure
def get_no_selection(m):
    # Solve the quadratic equation to find the selection number
    n = math.ceil((1 + math.sqrt(1+(8*m)))/2)
    return n

# Method to perform the selection procedure using robot fitness
def selection(robots):
    # Get number of robots to select
    no_selection = get_no_selection(len(robots))

    fitness_list = []
    selected = []

    for robot in robots:
        fitness_list.append(robot.fitness)
    
    # Pick the first few robots with the highest fitness values
    robots_copy = robots.copy()
    for i in range(no_selection):
        index = fitness_list.index(max(fitness_list))
        
        fitness_list.pop(index)
        robot = robots_copy.pop(index)
        
        selected.append(robot)
    
    # Sort the robots in the order of their fitness and return
    selected.sort(key=lambda x: x.fitness, reverse=True)

    return selected

# Method to perform the crossover operation on selected robots
def crossover(robots, pop_size):

    crossover_bots = []

    # Crossover all combinations of selected robots until 
    # the population size is recreated
    for i in range(len(robots)-1):
        for j in range(i, len(robots)):
            init_state = robots[i].init_state.copy()
            genome = RobotGenome.crossover_genomes(robots[i].genome, robots[j].genome)
            new_robot = Robot(genome, init_state)

            crossover_bots.append(new_robot)

            if(len(crossover_bots) == pop_size):
                break
        else:
            continue
        break

    return crossover_bots

# Method to perform the mutation operation
def mutate(robots):
    # With a certain probability and delta value, induce a mutation in the robot
    mutation_prob = 0.5
    mutation_delta = 0.5

    for robot in robots:
        if random.random() < mutation_prob:
            robot.genome.mutate(mutation_delta)

# Method to run the GAROBOT genetic algorithm approach
def garobot(pop_size, start, goal, config, anim_ax, show_animation=False, record_frequency=0.5):

    # Create object to record metrics
    ga_metric = GARobotMetrics()

    # Build initial population
    robots = []
    for i in range(pop_size):
        robots.append(Robot.create_robot(start))

    crossover_robots = robots

    recording_interval = round(config.num_gens * record_frequency)

    # For FIXED environments, build environment once with new obstacles
    if(config.fixed):
        env_config = EnvConfig(config.clutter_pct)

    # Run the genetic algorithm for the configured number of generations
    for num_gen in range(config.num_gens):
        
        robots = crossover_robots
        reached_bots = 0

        print('Generation',num_gen+1)
        for j in range(config.runs_per_gen):
            print('Run',j+1)

            # For GENERAL environments, build a new environment at every run
            if(not(config.fixed)):
                env_config = EnvConfig(config.clutter_pct)

            # Run the motions for the generation
            reached_bots += run_generation(robots, goal, env_config, anim_ax, show_animation)

        # At a given frequency, record the robot trajectories and obstacles for data
        if(record_frequency > 0 and num_gen % recording_interval == 0):
            ga_metric.record_robot_trajectories(robots, num_gen, env_config)

        # Evaluate the objective value of the population and record
        total_obj_val = evaluate(robots, config.robot_type_gains)

        ga_metric.record_metrics(robots, reached_bots, num_gen, config.runs_per_gen, total_obj_val)

        # Apply the GA parameters to the robots
        selected = selection(robots)
        crossover_robots = crossover(selected, pop_size)
        mutate(crossover_robots)

        print('Gen Cost:', total_obj_val)

        for robot in robots:
            robot.reset_fitness_params()

    # Record the resulting trajectories at the end
    ga_metric.record_robot_trajectories(robots, num_gen, env_config)
    
    print('GARobot done!')

    return robots, env_config, ga_metric
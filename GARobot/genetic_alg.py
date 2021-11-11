import math
import random

from controller import run_generation
from env_config import EnvConfig
from robot_factory import Robot, RobotGenome

goal_dist_gain = 1000

class RobotType:
    
    # Robot profiles and corresponding objective gains
    # [Collision, Time, Distance]
    SAFE = [100, 1, 1]
    FAST = [10, 10, 1]
    DIRECT = [10, 1, 10]

class GARobotConfig:
    def __init__(self):
        self.num_gens = 10
        self.runs_per_gen = 1
        self.robot_type_gains = RobotType.FAST

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

def get_no_reproduce(m):
    # solve the quadratic equation to find the selection number
    n = math.ceil((1 + math.sqrt(1+(8*m)))/2)
    return n

def reproduce(robots):
    # # Idk how to use this from the paper. Just gonna pick the best pop
    # selection_prob = 0.6 

    no_reproduce = get_no_reproduce(len(robots))

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
    
    selected.sort(key=lambda x: x.fitness, reverse=True)

    return selected

def crossover(robots, pop_size):
    # # Idk how to use this from the paper. Just gonna pick the best pop
    # crossover_prob = 0.6

    crossover_bots = []

    # Sort the input robots by fitness

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

def mutate(robots):
    mutation_prob = 0.5
    mutation_delta = 0.2

    for robot in robots:
        if random.random() <= mutation_prob:
            robot.genome.mutate(mutation_delta)

def garobot(pop_size, goal, config):

    # Build initial population
    robots = []
    gen_obj_vals = []

    for i in range(pop_size):
        robots.append(Robot.create_robot(0, 5))

    crossover_robots = robots

    # FIXED
    # Build environment with new obstacles
    env_config = EnvConfig()

    for i in range(config.num_gens):
        
        robots = crossover_robots

        print('Generation',i+1)
        for j in range(config.runs_per_gen):
            print('Run',j+1)

            # # VARIED
            # # Build environment with new obstacles
            # env_config = EnvConfig()

            # Run the motions for the generation
            run_generation(robots, goal, env_config, show_animation=False)

        # Evaluate the objective value of the population and record
        total_obj_val = evaluate(robots, config.robot_type_gains)
        gen_obj_vals.append(total_obj_val)

        # Apply the GA parameters to the robots
        selected = reproduce(robots)
        crossover_robots = crossover(selected, pop_size)
        mutate(crossover_robots)

        print('Gen Cost:', total_obj_val)

        for robot in robots:
            robot.reset_fitness_params()
    
    print('GARobot done!')

    return robots, env_config
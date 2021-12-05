"""
Controller to allow robots to navigate the cluttered environment

Author: Megnath Ramesh

"""

import numpy as np

import math
from plot_utils import animate

import motion_planner as plan

def run_generation(robots, goal, env_config, anim_ax, show_animation):
    '''
    Controller for all robots to traverse the environment from start to goal
    '''

    # Reset robot states and define experiment limit
    for robot in robots:
        robot.reset_robot_state()

    max_steps = 100 # Good value for this map
    time_limit_exceeded = False

    obs = env_config.obs
    i = 0
    
    # Run experiment for a maximum number of steps
    while i < max_steps:
        stopped_robots = 0
        for robot in robots:
            if not(time_limit_exceeded):
                if (robot.reached_goal):
                    # Check if a robot is stopped (reached goal)
                    stopped_robots += 1
                else:
                    # If robot is not at the goal, try to find a new trajectory for the robot
                    u, _, cost, no_collisions = plan.dwa_control(robot.state, robot, goal, obs)
                    robot.trajectory_cost += cost
                    robot.state = plan.motion(robot.state, u, robot.dt)

                    # Update the fitness parameters to compute the objective later
                    robot.update_fitness_metrics(u, no_collisions)

                    # Record trajectory
                    robot.trajectory = np.vstack((robot.trajectory, robot.state))

                    # Check if the robot has reached the goal
                    dist_to_goal = math.hypot(robot.state[0] - goal[0], robot.state[1] - goal[1])
                    robot.reached_goal = dist_to_goal <= 0.2
        
        # If animation is enabled, send to the animator code
        if show_animation and anim_ax is not None:
            animate(anim_ax, env_config, goal, robots, obs)
        
        # End simulator if all robots have reached the goal
        if(stopped_robots == len(robots)):
            print("Goal!!")
            break

        # Check if time limit is exceeded
        i += 1
        if(i >= max_steps):
            time_limit_exceeded = True

    # Update distance from goal for each robot if they did not reach the goal
    if(time_limit_exceeded):
        print("Time Limit Exceeded")
        for robot in robots:
            if(not(robot.reached_goal)):
                robot.trajectory_cost = float("inf")
                
                dist_from_goal = math.hypot(robot.state[0] - goal[0], robot.state[1] - goal[1])
                robot.distance_from_goal += dist_from_goal

    # Count and return the number of robots that have reached the goal
    reached_bots = 0
    for robot in robots:
        if(robot.trajectory_cost < float('inf')):
            reached_bots += 1

    return reached_bots

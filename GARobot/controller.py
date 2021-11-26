import numpy as np

import math
from plot_utils import animate

import motion_planner as plan

def run_generation(robots, goal, env_config, anim_ax, show_animation):
    
    for robot in robots:
        robot.reset_robot_state()

    max_steps = 100 # Good value for this map
    time_limit_exceeded = False

    obs = env_config.obs
    i = 0
    
    while i < max_steps:
        stopped_robots = 0
        for robot in robots:
            if not(time_limit_exceeded):
                if (robot.reached_goal):
                    stopped_robots += 1
                else:
                    u, predicted_trajectory, cost, no_collisions = plan.dwa_control(robot.state, robot, goal, obs)
                    robot.trajectory_cost += cost
                    robot.state = plan.motion(robot.state, u, robot.dt)

                    robot.update_fitness_params(u, no_collisions)

                    robot.trajectory = np.vstack((robot.trajectory, robot.state))

                    dist_to_goal = math.hypot(robot.state[0] - goal[0], robot.state[1] - goal[1])
                    robot.reached_goal = dist_to_goal <= 2 * robot.robot_radius
        
        if show_animation:
            animate(anim_ax, env_config, goal, robots, obs)
        
        if(stopped_robots == len(robots)):
            print("Goal!!")
            break

        i += 1

        if(i >= max_steps):
            time_limit_exceeded = True

    if(time_limit_exceeded):
        print("Time Limit Exceeded")
        for robot in robots:
            if(not(robot.reached_goal)):
                robot.trajectory_cost = float("inf")
                
                dist_from_goal = math.hypot(robot.state[0] - goal[0], robot.state[1] - goal[1])
                robot.distance_from_goal += dist_from_goal

    reached_bots = 0
    for robot in robots:
        if(robot.trajectory_cost < float('inf')):
            reached_bots += 1

    print('Robots reaching goal:', reached_bots)

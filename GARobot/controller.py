import numpy as np

import matplotlib.pyplot as plt

import math
from plot_utils import plot_obstacles, plot_robot

import motion_planner as plan

def run_generation(robots, goal, env_config, show_animation = False):
    
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
                    robot.reached_goal = dist_to_goal <= 2*robot.robot_radius
        
        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(goal[0], goal[1], "xb")
            plot_obstacles(obs)

            for robot in robots:
                plt.plot(robot.state[0], robot.state[1], "xr")
                plot_robot(robot.state[0], robot.state[1], robot.robot_radius)

            plt.xlim(env_config.env_range)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)
        
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

    plt.cla()
    plt.plot(goal[0], goal[1], "xb")
    plot_obstacles(obs)
    
    plt.xlim(env_config.env_range)
    plt.axis("equal")
    plt.grid(True)

    # i = 1
    # for robot in robots:

    #     # if(robot.trajectory_cost < float('inf')):
    #     #     print('Robot', i)
    #     #     print('Goal gain: ', robot.genome.to_goal_cost_gain)
    #     #     print('Obstacle gain: ', robot.genome.obstacle_cost_gain)
    #     #     print('Obstacle Sphere of Influence: ', robot.genome.obstacle_sphere_of_influence)
    #     #     print('Cost: ', robot.trajectory_cost)

    #     plt.plot(robot.state[0], robot.state[1], "xr")
    #     plt.plot(robot.trajectory[:, 0], robot.trajectory[:, 1], "-r")

    #     i += 1

    print("Done")

"""
Mobile robot motion planning sample with Dynamic Window Approach

Authors: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
Modified by Megnath Ramesh for ECE 750 Project

"""

import numpy as np
import matplotlib.pyplot as plt

import math
import time
from plot_utils import plot_obstacles, plot_robot

from env_config import EnvConfig

def dwa_control(x, config, goal, obs):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory, cost = calc_control_and_trajectory(x, dw, config, goal, obs)

    return u, trajectory, cost

def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x
    """

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          config.min_speed, config.max_speed]

    # Dynamic window from motion model
    Vd = [x[2] - config.max_accel * config.dt,
          x[2] + config.max_accel * config.dt,
          x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt]

    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw

def predict_trajectory(x_init, v_x, v_y, config):
    """
    predict trajectory with an input
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        x = motion(x, [v_x, v_y], config.dt)
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, obs):
    """
    calculation final input with dynamic window
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])

    # evaluate all trajectory with sampled input in dynamic window
    for v_x in np.arange(dw[0], dw[1], config.v_resolution):
        for v_y in np.arange(dw[2], dw[3], config.v_resolution):

            trajectory = predict_trajectory(x_init, v_x, v_y, config)
            # calc cost
            to_goal_cost = config.genome.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            ob_cost = config.genome.obstacle_cost_gain * calc_obstacle_cost(trajectory, obs, config)

            final_cost = to_goal_cost + ob_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v_x, v_y]
                best_trajectory = trajectory
    return best_u, best_trajectory, min_cost

def motion(x, u, dt):
    """
    motion model
    """

    x[0] += u[0] * dt
    x[1] += u[1] * dt
    x[2] = u[0]
    x[3] = u[1]

    return x

def calc_obstacle_cost(trajectory, obs, config):
    """
    calc obstacle cost 
    max_obs_cost: collision
    """
    max_obs_cost = 1000000
    if(obs.size == 0):
        return 0
    
    ox = obs[:, 0]
    oy = obs[:, 1]
    o_radius = obs[:, 2]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.asarray(np.hypot(dx, dy))
    min_r = np.min(r, axis=1) # Get min distance from each obstacle
    total_cost = 0
    for i in range(len(min_r)):
        least_dist = (config.robot_radius + o_radius[i])
        if min_r[i] <= least_dist:
            return max_obs_cost # collision
        if(min_r[i] <= config.genome.obstacle_sphere_of_influence):
            total_cost += 1/(min_r[i] - least_dist)
    return total_cost

def calc_to_goal_cost(trajectory, goal):
    """
    calc to goal cost with angle difference
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    cost = np.sqrt(dx**2 + dy**2)
    return cost

def run_generation(robots, goal, show_animation = False):
    
    max_steps = 300 # Good value for this map
    time_limit_exceeded = False

    env_config = EnvConfig()
    obs = env_config.obs
    i = 0
    
    while i < max_steps:
        stopped_robots = 0
        for robot in robots:
            if not(time_limit_exceeded):
                if (robot.is_running):
                    u, predicted_trajectory, cost = dwa_control(robot.state, robot, goal, obs)
                    robot.trajectory_cost += cost
                    robot.state = motion(robot.state, u, robot.dt)
                    robot.trajectory = np.vstack((robot.trajectory, robot.state))

                    dist_to_goal = math.hypot(robot.state[0] - goal[0], robot.state[1] - goal[1])
                    robot.is_running = dist_to_goal > robot.robot_radius
                else:
                    stopped_robots += 1
        
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
                plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")

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

    plt.cla()
    plt.plot(goal[0], goal[1], "xb")
    plot_obstacles(obs)
    
    plt.xlim(env_config.env_range)
    plt.axis("equal")
    plt.grid(True)

    i = 1
    for robot in robots:
        print('Robot', i, ': ')
        print('Goal gain: ', robot.genome.to_goal_cost_gain)
        print('Obstacle gain: ', robot.genome.obstacle_cost_gain)
        print('Obstacle Sphere of Influence: ', robot.genome.obstacle_sphere_of_influence)
        print('Cost: ', robot.trajectory_cost)

        plt.plot(robot.state[0], robot.state[1], "xr")
        plt.plot(robot.trajectory[:, 0], robot.trajectory[:, 1], "-r")

        i += 1

    print("Done")

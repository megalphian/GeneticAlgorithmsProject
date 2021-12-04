"""
Robot motion planning with Dynamic Window Approach

Authors: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
Modified by Megnath Ramesh for ECE 750 Project

"""

import numpy as np

def dwa_control(x, config, goal, obs):
    """
    Dynamic Window Approach control
    
    Authors: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
    Modified by Megnath Ramesh for ECE 750 Project
    """
    dw = calc_dynamic_window(x, config)

    return calc_control_and_trajectory(x, dw, config, goal, obs)

def calc_dynamic_window(x, config):
    """
    calculation dynamic window based on current state x

    Authors: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
    Modified by Megnath Ramesh for ECE 750 Project
    """

    # Dynamic window from robot specification
    dw = [config.min_speed, config.max_speed,
          config.min_speed, config.max_speed]

    return dw

def predict_trajectory(x_init, v_x, v_y, config):
    """
    predict trajectory with an input
    
    Authors: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
    Modified by Megnath Ramesh for ECE 750 Project
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

    Authors: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
    Modified by Megnath Ramesh for ECE 750 Project
    """

    x_init = x[:]
    min_cost = float("inf")
    best_u = [0.0, 0.0]
    best_trajectory = np.array([x])
    min_collisions = 0

    # evaluate all trajectory with sampled input in dynamic window
    for v_x in np.arange(dw[0], dw[1], config.v_resolution):
        for v_y in np.arange(dw[2], dw[3], config.v_resolution):

            # predict a trajectory in the dynamic window
            trajectory = predict_trajectory(x_init, v_x, v_y, config)
            # calculate cost
            to_goal_cost = config.genome.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            ob_cost, collisions = calc_obstacle_cost(trajectory, obs, config)

            final_cost = to_goal_cost + ob_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v_x, v_y]
                best_trajectory = trajectory
                min_collisions = collisions

    return best_u, best_trajectory, min_cost, min_collisions

def motion(x, u, dt):
    """
    motion model

    Authors: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
    Modified by Megnath Ramesh for ECE 750 Project
    """

    x[0] += u[0] * dt
    x[1] += u[1] * dt
    x[2] = u[0]
    x[3] = u[1]

    return x

def calc_obstacle_cost(trajectory, obs, config):
    """
    calc obstacle cost 
    max_obs_cost: collision with an obstacle
    Returns: (cost, no. of collisions)

    Authors: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
    Modified by Megnath Ramesh for ECE 750 Project
    """
    max_obs_cost = 100000
    collisions = 0
    if(obs.size == 0):
        return (0, 0)
    
    ox = obs[:, 0]
    oy = obs[:, 1]
    o_radius = obs[:, 2]
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]
    r = np.asarray(np.hypot(dx, dy)) - o_radius[:, None]
    min_r = np.min(r, axis=1)
    i = np.argmin(min_r) # Get the min distance of a trajectory from each obstacle

    cost = 0
    influence_dist = (config.genome.obstacle_sphere_of_influence + config.robot_radius)
    if min_r[i] < config.robot_radius:
        cost = max_obs_cost # collision
        collisions = 100
    elif(min_r[i] <= influence_dist):
        collisions = 1
        cost = config.genome.obstacle_cost_gain * (1/min_r[i] - 1/influence_dist)
    return (cost, collisions)

def calc_to_goal_cost(trajectory, goal):
    """
    calc cost to go to goal from current position

    Authors: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
    Modified by Megnath Ramesh for ECE 750 Project
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]
    cost = np.sqrt(dx**2 + dy**2)
    return cost
"""
Robot motion planning with Dynamic Window Approach

Original Source: https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/DynamicWindowApproach/dynamic_window_approach.py

Authors: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
Modified by Megnath Ramesh for ECE 750 Project
"""

import numpy as np

def dwa_control(x, config, goal, obs):
    """
    Control for Dynamic Window Approach
    
    Authors: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
    Modified by Megnath Ramesh for ECE 750 Project
    """
   
    # Dynamic window from robot specification
    
    ### Modifications by Megnath Ramesh
    dw = [config.min_speed, config.max_speed,
          config.min_speed, config.max_speed]
    ### End of modifications

    return calc_control_and_trajectory(x, dw, config, goal, obs)

def predict_trajectory(x_init, v_x, v_y, config):
    """
    Predict trajectory with the given velocity inputs
    
    Authors: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
    Modified by Megnath Ramesh for ECE 750 Project
    """

    x = np.array(x_init)
    trajectory = np.array(x)
    time = 0
    while time <= config.predict_time:
        
        ### Modifications by Megnath Ramesh
        x = motion(x, [v_x, v_y], config.dt)
        ### End of modifications
        
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory

def calc_control_and_trajectory(x, dw, config, goal, obs):
    """
    Calculate the final control input and trajectory given the dynamic window

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

            ### Modifications by Megnath Ramesh
            # Predict trajectory in the dynamic window given the inputs
            trajectory = predict_trajectory(x_init, v_x, v_y, config)
            # Calculate the cost of the trajectory
            to_goal_cost = config.genome.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            ob_cost, collisions = calc_obstacle_cost(trajectory, obs, config)

            final_cost = to_goal_cost + ob_cost
            ### End of modifications

            # If the trajectory has the minimum cost, store and return later
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v_x, v_y]
                best_trajectory = trajectory
                min_collisions = collisions

    return best_u, best_trajectory, min_cost, min_collisions

def motion(x, u, dt):
    """
    Robot Motion Model

    Authors: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
    Modified by Megnath Ramesh for ECE 750 Project
    """

    ### Modifications by Megnath Ramesh
    x[0] += u[0] * dt
    x[1] += u[1] * dt
    ### End of modifications

    x[2] = u[0]
    x[3] = u[1]

    return x

def calc_obstacle_cost(trajectory, obs, config):
    """
    Calculate the trajectory cost in the presence of obstacles in the environment.
    Also record the number of collisions in the trajectory.

    Authors: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
    Modified by Megnath Ramesh for ECE 750 Project
    """
    max_obs_cost = 100000
    collisions = 0
    if(obs.size == 0):
        return (0, 0)
    
    ox = obs[:, 0]
    oy = obs[:, 1]
    
    ### Modifications by Megnath Ramesh
    o_radius = obs[:, 2]
    ### End of modifications

    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]

    ### Modifications by Megnath Ramesh
    r = np.asarray(np.hypot(dx, dy)) - o_radius[:, None]
    ### End of modifications

    min_r = np.min(r, axis=1)
    i = np.argmin(min_r) # Get the min distance of a trajectory from each obstacle

    ### Modifications by Megnath Ramesh
    # Compute the highest cost due to obstacles in the trajectory predicted
    cost = 0
    influence_dist = (config.genome.obstacle_sphere_of_influence + config.robot_radius)
    # If the robot has collided with an obstacle, penalize the trajectory heavily.
    if min_r[i] < config.robot_radius:
        cost = max_obs_cost
        collisions = 100
    elif(min_r[i] <= influence_dist):
        collisions = 1
        cost = config.genome.obstacle_cost_gain * (1/min_r[i] - 1/influence_dist)
    ### End of modifications

    return (cost, collisions)

def calc_to_goal_cost(trajectory, goal):
    """
    Calculate the cost to go to the goal from the current position.

    Authors: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
    Modified by Megnath Ramesh for ECE 750 Project
    """

    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]

    ### Modifications by Megnath Ramesh
    cost = np.sqrt(dx**2 + dy**2)
    ### End of modifications
    
    return cost
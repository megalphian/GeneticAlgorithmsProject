"""

Mobile robot motion planning sample with Dynamic Window Approach

author: Atsushi Sakai (@Atsushi_twi), Göktuğ Karakaşlı
Modified by Megnath Ramesh for ECE 750 Project

"""

import math

import matplotlib.pyplot as plt
import numpy as np

import random
import time

show_animation = False


def dwa_control(x, config, goal, obs):
    """
    Dynamic Window Approach control
    """
    dw = calc_dynamic_window(x, config)

    u, trajectory, cost = calc_control_and_trajectory(x, dw, config, goal, obs)

    return u, trajectory, cost

def generate_random_obstacle(env_min, env_max, max_rad):
    x_pos = random.uniform(env_min + max_rad, env_max - max_rad)
    y_pos = random.uniform(env_min + max_rad, env_max - max_rad)
    rad = random.uniform(0.05, max_rad)
    area = np.pi * (rad ** 2)
    return ([x_pos, y_pos, rad], area)


class EnvConfig:
    def __init__(self, clutter_ratio = 0.1):
        self.env_range = [0, 30]
        self.no_obstacles = 0
        self.max_obs_radius = 0.5
        self.obs = []
        area = 15 ** 2
        obs_area = 0
        while obs_area < (area * clutter_ratio):
            temp_ob, ob_area = generate_random_obstacle(self.env_range[0] + 7.5, self.env_range[1] - 7.5, self.max_obs_radius)
            self.obs.append(temp_ob)
            obs_area += ob_area
        self.obs = np.array(self.obs)

class Config:
    """
    simulation parameter class
    """

    def __init__(self):
        # Parameters to tune for GARobot
        # Also used to check if goal is reached in both types
        self.to_goal_cost_gain = 5
        self.obstacle_cost_gain = 4
        self.obstacle_sphere_of_influence = 0.8 # [m] for obstacle potential field

        # robot parameter
        self.robot_radius = 0.05 # For collision check
        self.max_speed = 1 # [m/s]
        self.min_speed = -0.5  # [m/s]
        self.max_accel = 0.5  # [m/ss]
        self.v_resolution = 0.01  # [m/s]
        self.dt = 0.1  # [s] Time tick for motion prediction
        self.predict_time = 0.1 # [s]

config = Config()
env_config = EnvConfig()

def motion(x, u, dt):
    """
    motion model
    """

    x[0] += u[0] * dt
    x[1] += u[1] * dt
    x[2] = u[0]
    x[3] = u[1]

    return x


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
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, obs, config)

            final_cost = to_goal_cost + ob_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v_x, v_y]
                best_trajectory = trajectory
    return best_u, best_trajectory, min_cost


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
        if(min_r[i] <= config.obstacle_sphere_of_influence):
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

def plot_robot(x, y, radius):  # pragma: no cover
    circle = plt.Circle((x, y), radius, color="b")
    plt.gcf().gca().add_artist(circle)

def plot_obstacles(obs):  # pragma: no cover
    for ob in obs:
        circle = plt.Circle((ob[0], ob[1]), ob[2], color="k")
        plt.gcf().gca().add_artist(circle)

def main(sx=0.0, sy=15.0, gx=30.0, gy=15.0):
    print(__file__ + " start!!")
    # initial state [x(m), y(m), v_x(m/s), v_y(m/s)]
    x = np.array([sx, sy, 0, 0.0])
    x_1 = np.array([sx+0.5, sy-0.5, 0, 0.0])
    x_2 = np.array([sx+0.5, sy+0.5, 0, 0.0])

    # goal position [x(m), y(m)]
    goal = np.array([gx, gy])

    # input [forward speed, yaw_rate]
    trajectory = np.array(x)
    trajectory_1 = np.array(x_1)
    trajectory_2 = np.array(x_2)

    total_cost = 0
    total_cost_1 = 0
    total_cost_2 = 0

    stop_x = False
    stop_x_1 = False
    stop_x_2 = False

    start_time = time.time()
    time_limit = 30 # Good value for this map
    time_limit_exceeded = False

    obs = env_config.obs
    while True:
        
        if not(stop_x or time_limit_exceeded):
            u, predicted_trajectory, cost = dwa_control(x, config, goal, obs)
            total_cost += cost
            x = motion(x, u, config.dt)  # simulate robot
            trajectory = np.vstack((trajectory, x))  # store state history

        if not(stop_x_1 or time_limit_exceeded):
            u_1, predicted_trajectory_1, cost_1 = dwa_control(x_1, config, goal, obs)
            total_cost_1 += cost_1
            x_1 = motion(x_1, u_1, config.dt)  # simulate robot
            trajectory_1 = np.vstack((trajectory_1, x_1))  # store state history

        if not(stop_x_2 or time_limit_exceeded):
            u_2, predicted_trajectory_2, cost_2 = dwa_control(x_2, config, goal, obs)
            total_cost_2 += cost_2
            x_2 = motion(x_2, u_2, config.dt)  # simulate robot
            trajectory_2 = np.vstack((trajectory_2, x_2))  # store state history

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(goal[0], goal[1], "xb")
            plot_obstacles(obs)

            plt.plot(x[0], x[1], "xr")
            plot_robot(x[0], x[1], config.robot_radius)
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], "-g")

            plt.plot(x_1[0], x_1[1], "xr")
            plot_robot(x_1[0], x_1[1], config.robot_radius)
            plt.plot(predicted_trajectory_1[:, 0], predicted_trajectory_1[:, 1], "-g")

            plt.plot(x_2[0], x_2[1], "xr")
            plot_robot(x_2[0], x_2[1], config.robot_radius)
            plt.plot(predicted_trajectory_2[:, 0], predicted_trajectory_2[:, 1], "-g")

            plt.xlim(env_config.env_range)
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # check reaching goal
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        dist_to_goal_1 = math.hypot(x_1[0] - goal[0], x_1[1] - goal[1])
        dist_to_goal_2 = math.hypot(x_2[0] - goal[0], x_2[1] - goal[1])

        stop_x = dist_to_goal <= config.robot_radius
        stop_x_1 = dist_to_goal_1 <= config.robot_radius
        stop_x_2 = dist_to_goal_2 <= config.robot_radius

        if (stop_x and stop_x_1 and stop_x_2):
            print("Goal!!")
            break

        cur_time = time.time()
        time_limit_exceeded = True if cur_time - start_time > time_limit else False

        if(time_limit_exceeded):
            print("Time Limit Exceeded!")
            break

    print('Cost: ', total_cost)
    print('Cost_1: ', total_cost_1)
    print('Cost_2: ', total_cost_2)
    print("Done")
    
    plt.cla()
    plt.plot(goal[0], goal[1], "xb")
    plot_obstacles(obs)
    
    plt.xlim(env_config.env_range)
    plt.axis("equal")
    plt.grid(True)

    plt.plot(x[0], x[1], "xr")
    plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")

    plt.plot(x_1[0], x_1[1], "xr")
    plt.plot(trajectory_1[:, 0], trajectory_1[:, 1], "-r")

    plt.plot(x_2[0], x_2[1], "xr")
    plt.plot(trajectory_2[:, 0], trajectory_2[:, 1], "-r")

    plt.show()


if __name__ == '__main__':
    main()